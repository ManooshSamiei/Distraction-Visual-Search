DIR="/nas/EOS/users/manoosh/sal/"
PYSAL=False
#docker run --gpus device=5 --rm -u $(id -u):$(id -g) -v $(pwd):/workspace \
#    -v /mnt:/mnt -v /media:/media -v /srv:/srv -v /nas:/nas -p 6967:6967 eos/tf1.15-conda:latest \
python /nas/EOS/users/manoosh/sal/Predicting-Salience-During-Visual-Search/compute_saliency_metrics.py \
    --path=$DIR \
    --use-pysaliency=$PYSAL

