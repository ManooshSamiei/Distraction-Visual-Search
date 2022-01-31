DOWNLOAD_DIR="/home/manoosh.samiei/Desktop/VisualAttention"
SIGMA=11
DATA_DIR="/home/manoosh.samiei/Desktop/VisualAttention/cocosearch"

#docker run --gpus device=2 --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
#    -v /mnt:/mnt -v /media:/media -v /srv:/srv -v /nas:/nas -p 1967:1967 eos/tf1.15-conda:latest \

python /home/manoosh.samiei/PycharmProjects/pythonProject/data_preprocessing.py \
    --dldir=$DOWNLOAD_DIR \
    --sigma=$SIGMA \
    --datadir=$DATA_DIR \

