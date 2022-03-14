DIR="/nas/EOS/users/manoosh/sal/"
PHASE="test"
THRESHOLD=30

docker run --gpus device=4 --rm -u $(id -u):$(id -g) -v $(pwd):/home/eos/workspace \
    -v /mnt:/mnt -v /media:/media -v /srv:/srv -v /nas:/nas -p 7967:7967 eos/tf1.15-conda:latest_ss \
    python /nas/EOS/users/manoosh/sal/Predicting-Salience-During-Visual-Search/main.py \
        --path=$DIR \
        --phase=$PHASE \
        --threshold=$THRESHOLD

        