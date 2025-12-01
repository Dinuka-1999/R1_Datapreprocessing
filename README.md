# R1
The ```Image_Analysis``` folder contains important files for studying images prior to the Unet model.

```SegmentOverlay.ipynb``` - This file is designed to overlay the segmentation maps on top of the images using python napari. We visualise only the endocardium and myocardium channels. Each label is shown in separate layers. ```gen_map_labels_if_available/raw_map_labels_if_available``` are supposed to be identified automatically but given that maps are extremely large and it is time consuming to fin all the unique labels, we hard code it. 

```image_analysis.ipynb``` - This file has many subsections important to analyse images. 