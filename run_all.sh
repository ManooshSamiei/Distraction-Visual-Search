# rm -r /nas/EOS/users/manoosh/sal/cocosearch/*
# rm -r /nas/EOS/users/manoosh/sal/results/*

DOWNLOAD_DIR="/nas/EOS/users/manoosh/sal"

SIGMA=11
DATA_DIR="/nas/EOS/users/manoosh/sal/cocosearch"
AVAILABLE_GPU="0"
##preprocess data
# docker run --gpus all -e CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
#     -v /mnt:/mnt -v /media:/media -v /srv:/srv -v /nas:/nas -p 7967:7967 eos/tf1.15-conda:latest_ss \
#     python /nas/EOS/users/manoosh/sal/Predicting-Salience-During-Visual-Search/data_preprocessing.py \
#     --dldir=$DOWNLOAD_DIR \
#     --sigma=$SIGMA \
#     --datadir=$DATA_DIR \

##train model

DIR="/nas/EOS/users/manoosh/sal/"
PHASE="train"
THRESHOLD=30

# docker run --gpus all -e CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
#     -v /mnt:/mnt -v /media:/media -v /srv:/srv -v /nas:/nas -p 7967:7967 eos/tf1.15-conda:latest_ss \
#     python /nas/EOS/users/manoosh/sal/Predicting-Salience-During-Visual-Search/main.py \
#         --path=$DIR \
#         --phase=$PHASE \
#         --threshold=$THRESHOLD

##test model

PHASE_1="test"

# docker run --gpus all -e CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
#     -v /mnt:/mnt -v /media:/media -v /srv:/srv -v /nas:/nas -p 7967:7967 eos/tf1.15-conda:latest_ss \
#     python /nas/EOS/users/manoosh/sal/Predicting-Salience-During-Visual-Search/main.py \
#         --path=$DIR \
#         --phase=$PHASE_1 \
#         --threshold=$THRESHOLD

##compute saliency metrics

PYSAL=False
CSV_DIR=${DOWNLOAD_DIR}/"results/"

docker run --gpus all -e CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
    -v /mnt:/mnt -v /media:/media -v /srv:/srv -v /nas:/nas -p 7967:7967 eos/tf1.15-conda:latest_ss \
    python /nas/EOS/users/manoosh/sal/Predicting-Salience-During-Visual-Search/compute_saliency_metrics.py \
        --path=$DIR \
        --use-pysaliency=$PYSAL \
        --csv-path=$CSV_DIR