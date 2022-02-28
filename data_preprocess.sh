DOWNLOAD_DIR="/nas/EOS/users/manoosh/sal"
SIGMA=11
DATA_DIR="/nas/EOS/users/manoosh/sal/cocosearch"

#docker run --gpus device=2 --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
#    -v /mnt:/mnt -v /media:/media -v /srv:/srv -v /nas:/nas -p 1967:1967 eos/tf1.15-conda:latest \

python /nas/EOS/users/manoosh/sal/Predicting-Salience-During-Visual-Search/data_preprocessing.py \
    --dldir=$DOWNLOAD_DIR \
    --sigma=$SIGMA \
    --datadir=$DATA_DIR \

