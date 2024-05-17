#!/bin/bash

grep "Best Model results" logs/$1/$2/*.out > feat_prop_$1_$2_collected.out
