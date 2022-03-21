# Where people fixate when searching for an object

This repository contains the tensorflow implementation of paper **"Where people fixate when searching for an object"** 

## Architecture
<img src="./images/Model.png" width="800"/>

**_Abstract:_** *Most studies in computational modeling of visual attention are focused on task-free observation of images. However, free-viewing saliency considers very limited scenarios of humans’ behavior in daily life. Most visual activities that humans perform daily are goal-oriented and demand a great amount of top-down attention control. Visual Search is
a simple task that requires observers to look for a target object in a scene. Compared to free-viewing saliency, visual search demands
more top-down control of attention. In this paper, we adapt a light-weight free-viewing saliency model to model humans’ visual attention behavior during visual search. Our approach predicts fixation density maps, the probability of eye fixation over pixels of the search image, using a two-stream encoder-decoder network. This method helps us to predict which locations are more distracting when searching for a particular target. We use the cocosearch18 dataset to train and evaluate our model. Our network achieves noticeable results on the state-of-the-art saliency metrics (AUC-Judd=0.95, AUC-Borji=0.85, sAUC=0.84, NSS=4.64, KLD=0.93, CC=0.72, SIM=0.54, and IG=2.59).*

Original repository forked from the implementation of MSI-Net saliency network [Contextual encoder-decoder network for visual saliency prediction] (https://www.sciencedirect.com/science/article/pii/S0893608020301660) (2020), [original implementation](https://github.com/alexanderkroner/saliency)


## Citation

```
@article{Samiei2021visualsearch,
  title={Where people fixate when searching for an object},
    journal = {Journal of Vision},
    volume = {},
    number = {},
    pages = {},
    year = {},
    month = {},
    issn = {},
    doi = {},
    url = {},
    eprint = {},
}
```

## Requirements

| Package    | Version |
|:----------:|:-------:|
| python     | 3.7.6   |
| tensorflow | 1.15.0  |
| matplotlib | 3.1.3   |
| numpy      | 1.18.1  |
| cv2        | 4.3.0   |
| pandas     | 1.0.1   |
| gdown      | 4.4.0   |
| wget       | 3.2     |
| pysaliency | -       |

All dependencies can be installed in a single docker image or an environment. 

## Results

<img src="./images/results_1" width="1000"/>

## Dataset Download + Preprocessing

```dataset_download.py``` --> downloads COCO-Search18 dataset, target images, and VGG16 pretrained weights on ImageNet.

```data_preprocessing.py``` --> Creates task-image pairs. Processes fixation data and creates Gaussian-blurred fixation maps. It resizes all images and fixation maps, augments data with horizontal flips, splits augmented data into train-test-validation sets. Unblurred fixation maps are also generated for test split to be used in saliency metrics computation. 

 ```data_preprocessing.py``` file automatically calls ```dataset_download.py```. Thus, to download and do the preprocessing steps we only need to run ```data_preprocessing.py``` as below:

```
    python ./data_preprocessing.py \
    --dldir=$DOWNLOAD_DIR \
    --sigma=$SIGMA \
    --datadir=$DATA_DIR \
```

```sigma``` specifies the Gaussian blurring standard deviation. The default value of ```11``` is used. The choice of sigma <img src="https://render.githubusercontent.com/render/math?math= s = cutoff_frequency/np.sqrt(2*np.log(2))">  $s = cutoff_frequency/np.sqrt(2*np.log(2))$
## Running All Steps at Once

