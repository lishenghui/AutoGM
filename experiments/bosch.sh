#!/usr/bin/env bash


pushd ../ || exit

for method in Autogmfl Autogmpfl
do
for attack in data model
  do
    for pc in 0.0 0.1 0.2 0.3 0.4 0.5
    do
      for seed in 1 2 3 4 5
      do
          for lamb in 1000000000.0
          do
              python -u main.py --dataset bosch --model mlp --global_rounds 1000 --local_rounds 10 --scale \
              --num_gpus 2 --num_actors 10  -pc ${pc} --attack ${attack}  --lamb2 ${lamb} \
              --lr 0.01 --clients_per_round 5 --metrics_dir results --method ${method} --seed ${seed} --eval_every 10
          done
      done
    done
  done
done
popd || exit
