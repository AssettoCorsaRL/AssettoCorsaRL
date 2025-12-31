#!/bin/bash

python -m src.assettocorsa_rl.train.train \
    --total-steps 1000000 \
    --log-interval 1000 \
    --save-interval 50000 \
    --seed 6741 \
    --device cuda \
    --wandb-project AssetoCorsaRL-CarRacing