#!/bin/bash

grep "Best Model results" logs/$1/$2/$3/*.log > feat_prop_$1_$2_$3_collected.out
