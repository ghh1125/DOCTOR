#!/bin/bash


python main.py --dataset fakett --mode train --inference_ckp ./provided_ckp/FakingRecipe_fakett \
  --seed 2025 --dg --diffusion --alpha 0.1 --beta 3 --early_stop 15 --lr 1e-3 --gamma 0.05 --epoches 40 \
  --path_ckp './checkpoints1/' --path_tb "./tensorboard1/" --gpu 0


