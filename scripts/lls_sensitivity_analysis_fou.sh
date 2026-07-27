#!/bin/bash

python3 main_ablation_study.py \
-e gs_mars_earth_scenario_inc_4 gs_mars_earth_scenario_inc_8 gs_mars_earth_scenario_inc_12 gs_mars_earth_scenario_inc_16 \
-s lls \
--fou_min 250 --fou_max 3000 --fou_step 250