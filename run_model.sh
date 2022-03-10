DIR="/nas/EOS/users/manoosh/sal/"
PHASE="train"
THRESHOLD=30

docker run --gpus device=3 --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
    -v /mnt:/mnt -v /media:/media -v /srv:/srv -v /nas:/nas -p 7967:7967 eos/tf1.15-conda:latest \
    python /nas/EOS/users/manoosh/sal/Predicting-Salience-During-Visual-Search/main.py \
        --path=$DIR \
        --phase=$PHASE \
        --threshold=$THRESHOLD

        