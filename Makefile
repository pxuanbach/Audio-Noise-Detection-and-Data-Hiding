preprocessing:
	python ./datasets/preprocessing.py


train-master:
	python train_gan_distributed.py \
    --nodes 2 \
    --node-rank 0 \
    --master-addr 192.168.0.227 \
    --master-port 12355 \
    --batch-size 12 \
    --epochs 20 \
    --dataset-path datasets/processed


train-worker:
	python train_gan_distributed.py \
    --nodes 2 \
    --node-rank 1 \
    --master-addr 192.168.0.227 \
    --master-port 12355 \
    --batch-size 12 \
    --epochs 20 \
    --dataset-path datasets/processed
