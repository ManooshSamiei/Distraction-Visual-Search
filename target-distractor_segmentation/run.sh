DOWNLOAD_DIR="/home/manoosh.samiei/VisualSearch/"
AVAILABLE_GPU="0"
CATEGORY='bottle'#'car'#'bowl'#
CLASS=3
#download dataset
docker run --gpus all -e CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
    -v /mnt:/mnt -v /media:/media -v /nas:/nas -p 7167:7167 tf1.15-conda:latests \
    python /home/manoosh.samiei/VisualSearch/Distraction-Visual-Search/target-distractor_segmentation/download_data.py \
    --dldir=$DOWNLOAD_DIR 


#preprocess data
docker run --gpus all -e CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
    -v /mnt:/mnt -v /media:/media -v /nas:/nas -p 7457:7457 tf1.15-conda:latest \
    python /home/manoosh.samiei/VisualSearch/Distraction-Visual-Search/target-distractor_segmentation/data_preprocess.py \
    --dldir=$DOWNLOAD_DIR  \
    --category=$CATEGORY \
    --classification=$CLASS


#train and test the model
docker run --gpus all -e CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
    -v /mnt:/mnt -v /media:/media -v /nas:/nas -p 7197:7197 tf1.15-conda:latest \
    python /home/manoosh.samiei/VisualSearch/Distraction-Visual-Search/target-distractor_segmentation/main.py \
    --dldir=$DOWNLOAD_DIR \
    --classification=$CLASS 