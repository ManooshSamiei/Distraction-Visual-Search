rm -r /home/manoosh.samiei/VisualSearch/cocosearch/*
rm -r /home/manoosh.samiei/VisualSearch/results/*

##download dataset

AVAILABLE_GPU="0"
DOWNLOAD_DIR="/home/manoosh.samiei/VisualSearch/"

docker run --gpus all -e CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
    -v /mnt:/mnt -p 7967:7967 tf1.15-conda:latest \
    python /home/manoosh.samiei/VisualSearch/Distraction-Visual-Search/dataset_download.py \
    --dldir=$DOWNLOAD_DIR 


##preprocess data

SIGMA=11

docker run --gpus all -e CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
    -v /mnt:/mnt -p 7967:7967 tf1.15-conda:latest \
    python /home/manoosh.samiei/VisualSearch/Distraction-Visual-Search/data_preprocessing.py \
    --dldir=$DOWNLOAD_DIR \
    --sigma=$SIGMA 

##train model

PHASE="train"
THRESHOLD=30

docker run --gpus all -e CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
    -v /mnt:/mnt -p 7967:7967 tf1.15-conda:latest \
    python /home/manoosh.samiei/VisualSearch/Distraction-Visual-Search/main.py \
        --path=$DOWNLOAD_DIR \
        --phase=$PHASE \
        --threshold=$THRESHOLD

##test model

PHASE="test"

docker run --gpus all -e CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
    -v /mnt:/mnt -p 7967:7967 tf1.15-conda:latest \
    python /home/manoosh.samiei/VisualSearch/Distraction-Visual-Search/main.py \
        --path=$DOWNLOAD_DIR \
        --phase=$PHASE \
        --threshold=$THRESHOLD

##compute saliency metrics

PYSAL=False
CSV_DIR=${DOWNLOAD_DIR}"results/"

docker run --gpus all -e CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
    -v /mnt:/mnt -p 7967:7967 tf1.15-conda:latest \
    python /home/manoosh.samiei/VisualSearch/Distraction-Visual-Search/compute_saliency_metrics.py \
        --path=$DOWNLOAD_DIR  \
        --use-pysaliency=$PYSAL \
        --csv-path=$CSV_DIR