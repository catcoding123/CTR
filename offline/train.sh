#!/bin/bash
LOG_FILE=log/log.txt

python train_net.py \
      --task_type=train \
	  --learning_rate=0.0005 \
	  --optimizer=Adam \
	  --num_epochs=1 \
	  --batch_size=256 \
	  --field_size=39 \
	  --feature_size=117581 \
	  --deep_layers=400,400,400 \
	  --dropout=0.5,0.5,0.5 \
	  --log_steps=1000 \
	  --num_threads=8 \
	  --model_dir=./model_ckpt/criteo/DeepFM/ \
	  --data_dir=../../data/criteo/ \
	  --model_type=deepfm
      2>&1 | tee $LOG_FILE
