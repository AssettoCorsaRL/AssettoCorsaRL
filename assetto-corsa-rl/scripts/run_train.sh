#!/bin/bash

python -m src.assettocorsa_rl.train.train \
    --env-config-path src/assetto_corsa_rl/env/configs/env_config.yaml \
    --sac-config-path src/assetto_corsa_rl/model/configs/sac_config.yaml \
    --log-dir logs/train_runs \
    --total-timesteps 1000000 \
    --eval-interval 10000 \
    --save-interval 50000