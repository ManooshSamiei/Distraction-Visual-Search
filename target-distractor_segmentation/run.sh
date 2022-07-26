DOWNLOAD_DIR="/nas/EOS/users/manoosh/sal"
OUT_DIR="/nas/EOS/users/manoosh/sal/segmentation_results"
AVAILABLE_GPU="5"
CATEGORY='bowl' #'car'#'bowl'#
CLASS=2
# #download dataset
# docker run --gpus all -e CUDA_VISIBLE_DEVICES=$AVAILABLE_GPU --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
#     -v /mnt:/mnt -v /media:/media -v /srv:/srv -v /nas:/nas -p 7167:7167 eos/tf1.15-conda:latest_ss \
#     python /nas/EOS/users/manoosh/sal/Predicting-Salience-During-Visual-Search/target-distractor_segmentation/download_data.py \
#     --dldir=$DOWNLOAD_DIR 

# preprocess data
# docker run --gpus  '"device=0,1"' --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
#     -v /mnt:/mnt -v /media:/media -v /srv:/srv -v /nas:/nas -p 7457:7457 eos/tf1.15-conda:latest_ss \
#     python /nas/EOS/users/manoosh/sal/Predicting-Salience-During-Visual-Search/target-distractor_segmentation/data_preprocess.py \
#     --dldir=$DOWNLOAD_DIR  \
#     --category=$CATEGORY \
#     --classification=$CLASS

#train and test the model
docker run --gpus '"device=5"' --rm -u $(id -u):$(id -g) -v $(pwd):/home/eos/workspace \
    -v /mnt:/mnt -v /media:/media -v /srv:/srv -v /nas:/nas -p 4517:4517 eos/tf1.15-conda:latest_ss \
    python /nas/EOS/users/manoosh/sal/Predicting-Salience-During-Visual-Search/target-distractor_segmentation/main.py \
    --dldir=$DOWNLOAD_DIR \
    --classification=$CLASS \
    --outdir=$OUT_DIR \
    --category=$CATEGORY
