DOWNLOAD_DIR="/nas/EOS/users/manoosh/sal"
SIGMA=11
DATA_DIR="/nas/EOS/users/manoosh/sal/cocosearch"

python /nas/EOS/users/manoosh/sal/Predicting-Salience-During-Visual-Search/data_preprocessing.py \
    --dldir=$DOWNLOAD_DIR \
    --sigma=$SIGMA \
    --datadir=$DATA_DIR \

