#!/bin/bash

python3 main_ablation_study.py \
-e dte_mars_earth_scenario_inc_8 dte_mars_earth_scenario_inc_12 dte_mars_earth_scenario_inc_16 \
-s otls \
--alpha_min 0 --alpha_max 100 --alpha_step 1