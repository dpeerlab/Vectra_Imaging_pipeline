# Documentation for Vectra Analysis Pipeline

Cancer progression and metastasis involve complex interplay between heterogeneous cancer cells and the tumor microenvironment (TME). The current advances in cancer therapeutic options including immune-checkpoint inhibitors, novel targeted agents have emphasized the prognostic and pathophysiological roles of the TME. Multiplexed imaging data is providing dramatic opportunities to understand the tumor microenvironment, but there is an acute need for better analysis tools. Here, we provide a pipeline for multiplexed imaging quality control and processing. It contains three core steps: 

1. Preprocess raw images to remove undesired noise (introduced by technical sources) while retaining biological signal  
2. Perform segmentation to draw boundaries around individual cells, making it possible to discern morphology and which features, such as detected RNA or protein, belong to each cell.  
3. Extract cellular feature from images via segmentation and assign cell types to each cell. 

To transform digital images into cell-level measurements, we have been applying, comparing and optimizing cutting-edge computer vision and machine learning techniques to each of the steps above. All code is written in Python.



## Installation 

* Clone the GitHub [repository](https://github.com/dpeerlab/)

* Create a new conda environment for this pipeline 

* ```bash
  conda env create -n davinci -f environment.yml 
  source activate davinci
  python -m ipykernel install --user --name davinci
  ```

* Install this package

* ```bash
  python setup.py install
  ```

* Install the deep learning model for segmentation following the instruction on [link](https://github.com/dpeerlab/Mask_R-CNN_cell)



## Usage

### 0. Data inspection

The first step in the analysis is to inspect the experimental setup and examine the data. We will confirm the data composition, signal distribution and visualize the images into RGB (Red-Green-Blue) space.

#### 0.1. Image file inspection [Notebook](./notebook/0.1.Image_file_inspection.ipynb)

First, we check the composition of the image files, including information like sample id, image size and markers in the studies. Then we will remove the samples that are not included in a specific study, check the missing samples, and remove the duplicated images. 

Step 1: Input the path of the raw TIF data and use `glob` to finds all the pathnames matching a specified pattern according to the rules used by the Unix shell. Example:

```python
folder_path = '../data/tif/*.tif'
file_list = sorted(glob.glob(folder_path))
```

Step 2: Open the image files in a loop and save its size information. With the `np.unique` function, we will get the count of different image size. This step provide the important information of the image size composition in the data. Together with the magnification information, this will help understand the physical size of the image. Sometimes in one cohort, the image have different sizes, which is important to consider during the downstream analysis.

Step 3: Parse the marker information from the file metadata. If the information is stored during data generation, we can parse it out from metadata in the tiff file with `tifffile`. Otherwise, type the marker information there. It is very important to make sure the marker info is correct. It would be wise to check the actual image of each marker to confirm this.

Step 4: Check missing and redundant data in the dataset. Given a cohort, sometimes there are extra, redundant and missing samples in the dataset. Here, we first define a cohort id list, either from manual typing, or from other dataset. Then we parse the sample id from the image filename to identify extra, redundant and missing samples. At the end, we move redundant data with `shutil.move`, and try to find the missing data by contacting the ones who provide data.

Step 5. Check the data composition again after the correction.

#### 0.2. Image data inspection [Notebook](./notebook/0.2.Image_data_inspection.ipynb)

In a multiple sample experimental setup, data normalization is very important. In this step, we will look at the signal intensity distribution and find optimized parameters for normalization. Given the nature of the fluorescent imaging, the maximum intensity of the signal in the data is important to us.

Step 1. Get the image file list with `glob`, load the image with `tifffile.imread` and smooth the data with `ndimage.gaussian_filter` of scipy to get rid of outliers. 

Step 2. Display the distribution of max intensity from each image and each sample to have an overall impression of each channel. This step will provide us information like: what does the max intensity look like in marker positive images; what does it look like in negative ones; how different max values in positive images differ from each other. 

Step 3. Find a threshold for a marker being positive. One key note here is that not all images are positive for all markers. It is possible that there is no positive staining of a certain marker in some images. However, in most normalization cases, the image signals are usually normalized by max value in the image. To properly normalize these cases, we need to manually find a threshold for a marker being positive. And save the results together with the ones that are above this threshold. What's more, we visualize some of the images before and after normalization to validate the choice of the threshold.    

#### 0.3. Image visualization  [Notebook](./notebook/0.3.Image_visualization.ipynb)

In raw images from the imaging system are usually in a format with multiple single channel image. To better visualize them in one image, we need to convert some of the marker into RGB/multiple color space.

Step 1. Get the list of target `tif` images and design a color panel for the marker. In theory and in practice, 7 colors are the limit of a good visualization with them being `green, white(gray), cyan, yellow, magenta,blue, red  ` . The channel order for the color can be assigned in the function `convert_one_image` with parameter `color_list`. 

The `tqdm` function will show the progress of converting the images. Then when it is done, we can visualize them in notebook, or just open in windows. The images are saved in `png` format.



### 1. Preprocessing 

In this step, we will demonstrate preprocessing step, which is to remove undesired noise from the data. Then we will visualize before and after preprocessing as a quality control report.

#### 1.1. Image preprocessing  [Notebook](./notebook/1.1.Image_preprocessing.ipynb)

Here we will remove noise in the images. The thresholding can be done automatically. However, it does not always work. In this case, we will set a boundary for the thresholds per sample. We visualize the results here and record the min and max for the parameters if the automatical calculated values that based on the distribution are not perfect. Then at the end, we record the thresholds with the guidance of the min and max values.

Step 1. Create blank DataFrame to store the min and max threshold information in a per sample and per marker format. 

Step 2. Go through each channel and each sample, and set the best min, and max value for them. Most of the cases, `min=2`, `min=4` works. However, when the noises are strong, the values should be increased. 

Step 3. If the automatical thresholding function does not provide good values, check the `threshold_one`,`threshold_one_plot` function. 

#### 1.2. Image QC report  [Notebook](./notebook/1.2.Image_QC_report.ipynb)

The raw data of imaging systems are not human-readable. Even after converting them into RGB format, there are too many of them (hundreds) to watch one by one. The function of the QC report is to place the image from one sample into one file and display all the channels there before and after the preprocessing. Based on the report, we can fine-tune some preprocessing parameters or delete some bad data from the dataset.

Step 1. Load raw image, choose a report folder and project name. 

Step2. Generate a report just for raw data. Since we will parse the sample ID and other information from the file name, some future modification might be needed for different naming system. The user should find the `one_report` function from the source code and do some modification.

Step 3. Generate a report with before and after preprocessing images. Load the `threshold` and `max_cap` parameters from previous notebooks. Otherwise it is the same as the above step.

Step 4. Run through all the images and save the thresholds into a local file.

### 2. Segmentation 

To get single cell information from the images, we will use deep learning method - Mask R-CNN to do instance segmentation on the RGB images. Before the training, check the background of the method and the installation information via [link](https://github.com/dpeerlab/Mask_R-CNN_cell). GPU is highly recommended for training. 

#### 2.1. Train deep learning model on custom data [Notebook](./notebook/2.1.Image_segmentation_train.ipynb)

With target images and some human annotated masks, we can train the deep learning model to predict segmentation results. We recommend 10+ training images with size larger than 400x400 pixel under 200X magnification. 

Step 1.  Load the training images, and its annotation. Here we assume the mask is a single channel image with each object separated with each other. We will visualize 5 of them to check if the image and annotation match each other. By default, the images will be randomly cropped into 128*128 size during training. It provides best results during testing. You can change it in `CellConfig()`function. You can adjust the parameter `RPN_ANCHOR_SCALES`, which is the size of your target.

Step 2. Set the training configuration. `init_with = "imagnet" ` or `init_with = "coco" ` will largely improve the results since the model weighted will learn basic segmentation on natural images. We also add data  augmentation into our training as shown here.

```python
augmentation = iaa.SomeOf((0, 5), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Affine(rotate=(-90,90)),
    iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0.0,5.0)))
    ,iaa.Sometimes(0.2, iaa.Affine(scale={"x": (0.7, 1.5), "y": (0.7, 1.5)})),
    iaa.Sometimes(0.2,  iaa.Multiply((0.7, 1.5))),
    iaa.Sometimes(0.1, iaa.Affine(shear=(-45, 45)))
])
```

Step 3. During training, we will start with 10 epochs with 1000 steps per epoch (you can adjust the steps in `CellConfig()`) on the heads layers of the model. Then we will do another 20 epochs on all layers. At the end, another 10 epoch on all layers with smaller learning rate. It usually takes less than 1 day to finish the training.  

#### 2.2. Predict segmentation with pre-trained model [Notebook](./notebook/2.2.Image_Segmentation_prediction.ipynb)

We will show how to use pre-trained Mask R-CNN model to do prediction on your own dataset in this section. The training images and the prediction images do not have to be the same dataset. As long as they share similar pattern. For example, if cells and nuclei in both images are of similar size and shape, the tumor and immune cells have same coloring, then the prediction would work.

This step is recommended with CPU, actually, which is cheaper and provides more memories.

Step 1.  Load the path of the pre-trained weights.

Step 2. Set the inference parameters. `DETECTION_MAX_INSTANCES` and `POST_NMS_ROIS_INFERENCE` decide the max number of objects to be predicted in the image. The number of objects depends on the size of image. Smaller parameters allow faster prediction but you will want large enough value for big image. `DETECTION_MIN_CONFIDENCE` is the threshold for the confidence of the prediction. `DETECTION_NMS_THRESHOLD` controls the overlap level on the prediction. 

Step 3. Do prediction on training images to see if your training works.

Step 4. Do prediction on some validation images.

Step 5. Do prediction on the list of all images. And since some images can be too big to be predicted at once, we set an unit prediction size, e.g., 512*7104, and then do stitching on the results. 



### 3. Cell typing

With the cleaned, and normalized the signal and segmented cells, we can get spatial, morphology and expression information from the images for each cell. Then with these information, we can assign the cell one of the cell types.

#### 3.1. Cell feature extraction [Notebook](./notebook/3.1.Cell_feature_extraction.ipynb)

Step 1. Load all the tif, and segmentation files. Parse their unique ID and check if the order of files matches. 

Step 2. Load thresholding and cap dataset for preprocessing and normalization. 

Step 3. Run the feature extraction process. We use `parallel` function to utilize multiple CPU for this task. 

Step 4. Change the data type for better storage. Detailed feature and datatype information:

```
id_image             5145 non-null int16
id_sample            5145 non-null int16
id_cell              5145 non-null int32
area                 5145 non-null int16
centroid_x           5145 non-null float32
centroid_y           5145 non-null float32
orientation          5145 non-null float16
eccentricity         5145 non-null float16
minor_axis_length    5145 non-null float16
major_axis_length    5145 non-null float16
chan1_sum            5145 non-null float16
chan1_area           5145 non-null int16
chan2_sum            5145 non-null float16
chan2_area           5145 non-null int16
chan3_sum            5145 non-null float16
chan3_area           5145 non-null int16
chan4_sum            5145 non-null float16
chan4_area           5145 non-null int16
chan5_sum            5145 non-null float16
chan5_area           5145 non-null int16
chan6_sum            5145 non-null float16
chan6_area           5145 non-null int16
chan7_sum            5145 non-null float16
chan7_area           5145 non-null int16
```

#### 3.2. Cell typing [Notebook](./notebook/3.2.Cell_typing.ipynb)

Here we will inspect the distribution of cellular features and do cell typing. Then we visualize the cell in each cluster to tune the cell typing parameters.

Step 1. Load the feature DataFrame. 

Step 2. Display the distribution of the sum, mean and area of the signals. 

Step 3. Based on the distribution, find a cutoff for the average signal and the area of signals. After that, based on the co-expression pattern, allow only one of the expression be positive based on the property of the markers. E.g., for strong T cell expression with tumor signals, it is likely a T cell in tumor region. So we will assign it as a T cell and remove tumor expression.

Step 4. Visualize the positive example of each channel. And re-tune the cell typing parameters based on the visualization results.