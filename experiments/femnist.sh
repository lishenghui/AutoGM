#!/usr/bin/env bash


# export CUDA_VISIBLE_DEVICES=1


pushd ../ || exit


for method in Autogmpfl
do
  for attack in model
  do
    for pc in 0.0 0.1 0.2 0.3
    do
      for seed in 1 2 3 4 5
      do
          for lamb in 100000000.0
          do
              python -u main.py --global_rounds 600 --scale --num_gpus 2 -pc ${pc} --attack ${attack}  --lamb2 ${lamb} \
              --lr 0.025 --clients_per_round 16 --num_actors 16 --metrics_dir results --method ${method} --seed ${seed} --eval_every 20
          done
      done
    done
  done
done
