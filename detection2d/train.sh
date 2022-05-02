#!/bin/bash
if [ $FINETUNE == 1 ]
then
  echo "Finetuning with real data"
  python3 ./tools/train.py configs/faster_rcnn_finetune_config.py
else
  echo "Training base model with synthetic data"
  python3 ./tools/train.py configs/faster_rcnn_config.py
fi
