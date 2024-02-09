#!/bin/bash

strategies=(zeros mean random feat_prop)
sampled_datasets=(sampled_10
                  sampled_20
                  sampled_30
                  sampled_40
                  sampled_50
                  sampled_60
                  sampled_70
                  sampled_80
                  sampled_90)
random=(1 2 3 4 5)
prop_layers=(1 2 3 20)

for str in ${strategies[@]};
do
  for sam in ${sampled_datasets[@]};
  do
    for ran in ${random[@]};
    do
      echo $str $sam $ran
      if [[ $str == feat_prop ]]
      then
        for pr in ${prop_layers[@]};
        do
          python3.8 main.py --dataset $1 --strategy $str --feat_prop co --prop_layers $pr --masked_items_image ./data/$1/$sam"_"$ran.txt --masked_items_text ./data/$1/$sam"_"$ran.txt
        done
      else
        python3.8 main.py --dataset $1 --strategy $str --masked_items_image ./data/$1/$sam"_"$ran.txt --masked_items_text ./data/$1/$sam"_"$ran.txt
      fi
    done
  done
done
