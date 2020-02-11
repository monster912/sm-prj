#!/usr/bin/env bash
# Hynix: Pi
for seed in 0
do
python main.py --dataset 'hynix' --whiten_norm 'norm' --augment_mirror True --augment_translation 2 --n_labeled 476 --lr_max 0.003 --ratio_max 300.0 -e 100 --random_seed ${seed}
done