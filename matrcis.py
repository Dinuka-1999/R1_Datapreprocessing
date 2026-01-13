'''
What does this script do?
This script computes segmentation metrics (Dice and IoU) between reference and predicted segmentation files.
It supports per-slice metric computation and can handle multiple labels/regions.

'''
import numpy as np
import os
import torch 
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, save_json, load_json, isfile
import SimpleITK as sitk
import multiprocessing
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mpl    
import math

mpl.rcParams['font.family'] = 'DejaVu Serif'
mpl.rcParams['font.size'] = 12

image_map = {'0001': "15Ps", '0002': "E9.5", '0003': "8Ps", '0004': "13Ps", '0005': "10Ps", '0006': "19Ps",}

def load_segmentation(segmentation_path):
    seg = sitk.ReadImage(segmentation_path)
    npy_image = sitk.GetArrayFromImage(seg)
    if npy_image.ndim == 2:
        npy_image = npy_image[None, None]
    elif npy_image.ndim == 3:
        npy_image = npy_image[None]
    elif npy_image.ndim == 4:
        pass
    return npy_image.astype(np.float16)

def region_or_label_to_mask(segmentation: np.ndarray, region_or_label) -> np.ndarray:
    if np.isscalar(region_or_label):
        return segmentation == region_or_label
    else:
        mask = np.zeros_like(segmentation, dtype=bool)
        for r in region_or_label:
            mask[segmentation == r] = True
    return mask

def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def compute_metrics(reference_file: str, prediction_file: str,
                    labels_or_regions,folder_path:str,
                    ignore_label: int = None, per_slice: bool=False) -> dict:
    # load images
    seg_ref = load_segmentation(reference_file)
    seg_pred= load_segmentation(prediction_file)
    print(seg_ref.shape, seg_pred.shape)
    print("Files are loaded", reference_file)
    label_names = ['background',"Myocardium","Endocardium","Lumen","ECM"]
    ignore_mask = seg_ref == ignore_label if ignore_label is not None else None

    results = {}
    results['reference_file'] = reference_file
    results['prediction_file'] = prediction_file
    results['metrics'] = {}
    for r in labels_or_regions:
        results['metrics'][r] = {}
        mask_ref = region_or_label_to_mask(seg_ref, r)
        mask_pred = region_or_label_to_mask(seg_pred, r)

        if per_slice:
            dice_scores = []
            iou_scores = []
            img_dimensions = mask_ref.shape[1]
            
            for dim in range(img_dimensions):
                tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref[:,dim,:,:], mask_pred[:,dim,:,:], ignore_mask[:,dim,:,:] if ignore_mask is not None else None)
                if tp + fp + fn == 0:
                    dice = np.nan
                    iou = np.nan
                else:
                    dice = 2 * tp / (2 * tp + fp + fn)
                    iou = tp / (tp + fp + fn)
                dice_scores.append(dice)
                iou_scores.append(iou)
            # print(len(dice_scores), len(iou_scores))
            results['metrics'][r]['Dice_through_z_axis'] = dice_scores
            results['metrics'][r]['IoU_through_z_axis'] = iou_scores
            # save the plots
            plt.figure(figsize=(10, 5))
            plt.plot(dice_scores, label='Dice Score', marker='.')
            plt.plot(iou_scores, label='IoU', marker='.')
            plt.title(f'{label_names[r]} of {image_map[reference_file.split("/")[-1].split("_")[2].split(".")[0]]}')
            plt.xlabel('Slice Index')
            plt.ylabel('Metric Value')
            plt.ylim(0, 1)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path,"Eval_plots",f'{reference_file.split("/")[-1].split("_")[2].split(".")[0]}_{r}.png'),dpi=300)

        tp, fp, fn, tn = compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask)
        if tp + fp + fn == 0:
            results['metrics'][r]['Dice'] = np.nan
            results['metrics'][r]['IoU'] = np.nan
        else:
            results['metrics'][r]['Dice'] = 2 * tp / (2 * tp + fp + fn)
            results['metrics'][r]['IoU'] = tp / (tp + fp + fn)
        results['metrics'][r]['FP'] = fp
        results['metrics'][r]['TP'] = tp
        results['metrics'][r]['FN'] = fn
        results['metrics'][r]['TN'] = tn
        results['metrics'][r]['n_pred'] = fp + tp
        results['metrics'][r]['n_ref'] = fn + tp
    return results


def save_summary_json(results: dict, output_file: str):
    """
    stupid json does not support tuples as keys (why does it have to be so shitty) so we need to convert that shit
    ourselves
    """
    results_converted = deepcopy(results)
    # convert keys in mean metrics
    results_converted['mean'] = {str(k): results['mean'][k] for k in results['mean'].keys()}
    # convert metric_per_case
    for i in range(len(results_converted["metric_per_case"])):
        results_converted["metric_per_case"][i]['metrics'] = \
            {str(k): {metric:np.float64(results["metric_per_case"][i]['metrics'][k][metric])\
                      if metric!= 'Dice_through_z_axis'and metric!= 'IoU_through_z_axis' else results["metric_per_case"][i]['metrics'][k][metric]\
                        for metric in results["metric_per_case"][i]['metrics'][k].keys()}
             for k in results["metric_per_case"][i]['metrics'].keys()}
    # sort_keys=True will make foreground_mean the first entry and thus easy to spot
    save_json(results_converted, output_file, sort_keys=True)


def compute_metrics_on_folder(folder_ref: str, folder_pred: str, output_file: str,
                              per_slice,
                              file_ending,
                              regions_or_labels,
                              ignore_label: int = None,
                              num_processes: int = 8,
                              chill: bool = True) -> dict:
    
    if output_file is not None:
        assert output_file.endswith('.json'), 'output_file should end with .json'
    files_pred = subfiles(folder_pred, suffix=file_ending, join=False)
    files_ref = subfiles(folder_ref, suffix=file_ending, join=False)
    if not chill:
        present = [isfile(join(folder_pred, i)) for i in files_ref]
        assert all(present), "Not all files in folder_pred exist in folder_ref"
    files_ref = [join(folder_ref, i) for i in files_pred]
    files_pred = [join(folder_pred, i) for i in files_pred]
    
    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = pool.starmap(
            compute_metrics,
            list(zip(files_ref, files_pred, [regions_or_labels] * len(files_pred),[folder_pred]*len(files_pred),
                     [ignore_label] * len(files_pred), [per_slice] * len(files_pred)))
        )

    # mean metric per class
    metric_list = list(results[0]['metrics'][regions_or_labels[0]].keys())
    if per_slice:
        metric_list.remove('Dice_through_z_axis')
        metric_list.remove('IoU_through_z_axis')
        
    means = {}
    for r in regions_or_labels:
        means[r] = {}
        for m in metric_list:
            means[r][m] = np.nanmean([i['metrics'][r][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)

    label_names = ['background',"Myocardium","Endocardium","Lumen","ECM"]
    start_end = {'1':[23,549],'2':[0,697],'3':[21,317],'4':[30,375],'5':[13,367],'6':[19,595]}

    for ID,label in enumerate(regions_or_labels):
        Dice=[]
        IoU=[]
        max_length =0
        img_IDs=[]
        for imgs in results:
            Dice.append(imgs['metrics'][label]['Dice_through_z_axis'])
            IoU.append(imgs['metrics'][label]['IoU_through_z_axis'])
            max_length = max(max_length,len(imgs['metrics'][label]['Dice_through_z_axis']))
            img_IDs.append(imgs['reference_file'].split("/")[-1].split("_")[2].split(".")[0])
        
        fig1,ax1 = plt.subplots(figsize=(10,5))
        fig2,ax2 = plt.subplots(figsize=(10,5))
        
        for r in range(len(Dice)):
            n = len(Dice[r])
            offset = (max_length - n) // 2
            x = np.arange(offset, offset + n)  # center the shorter curve
            ax1.plot(x,Dice[r], marker='.',label=f'Image {img_IDs[r]}')
            ax1.scatter([offset+start_end[str(int(img_IDs[r]))][0], offset+start_end[str(int(img_IDs[r]))][1]], \
                        [Dice[r][start_end[str(int(img_IDs[r]))][0]],Dice[r][start_end[str(int(img_IDs[r]))][1]]], color='black', s=100)
            
            ax2.plot(x,IoU[r], marker='.',label=f'Image {img_IDs[r]}')
            ax2.scatter([offset+start_end[str(int(img_IDs[r]))][0], offset+start_end[str(int(img_IDs[r]))][1]], \
                        [IoU[r][start_end[str(int(img_IDs[r]))][0]],IoU[r][start_end[str(int(img_IDs[r]))][1]]], color='black', s=100)
        ax1.set_title(f'Dice through z-axis for {label_names[ID]}')
        ax1.set_xlabel('Z-axis (slices)')
        ax1.set_ylabel('Dice Score')
        ax1.legend()
        ax2.set_title(f'IoU through z-axis for {label_names[ID]}')
        ax2.set_xlabel('Z-axis (slices)')
        ax2.set_ylabel('IoU Score')
        ax2.legend()
        fig1.savefig(os.path.join(folder_pred,"Eval_plots",f'All_Dice_through_z_axis_label_{label}.pdf'),dpi = 300)
        fig2.savefig(os.path.join(folder_pred,"Eval_plots",f'All_IoU_through_z_axis_label_{label}.pdf'),dpi = 300)
        plt.close(fig1)
        plt.close(fig2)

    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    if output_file is not None:
        save_summary_json(result, output_file)
    else:
        print("No output file specified, not saving results.")

if __name__ == "__main__":

    Val_Folder = "/home/cellsmb/Desktop/Dinuka/Image_Analysis/Model_results/Model_results/IF_352_6img_results/nnUNetTrainerUMambaEncNoAMP__nnUNetPlans__2d/AllSegs"
    GT_Folder = "/home/cellsmb/Desktop/Dinuka/Image_Analysis/nnUnet_raw/labelsTr"

    if not os.path.exists(os.path.join(Val_Folder,"Eval_plots")):
        os.mkdir(os.path.join(Val_Folder,"Eval_plots"))
        
    compute_metrics_on_folder(
        folder_pred=Val_Folder,
        folder_ref=GT_Folder,
        output_file=join(Val_Folder, "Evaluation_summary_perslice.json"),
        per_slice=True,
        file_ending=".nii.gz",
        regions_or_labels=[0, 1, 2, 3, 4], # Check the compute matrics function for label mapping. Files copied from Linux desktop have labels starting from 1
        # labels or labelsTr were changed to 0-4, coppied from the mac
        ignore_label=None,
        num_processes=8,
        chill=True
    )
