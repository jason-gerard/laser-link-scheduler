#!/bin/bash

python3 main.py \
-e gs_mars_earth_scenario_inc_4 gs_mars_earth_scenario_inc_12 \
gs_mars_earth_scenario_inc_20 gs_mars_earth_scenario_inc_28 \
gs_mars_earth_scenario_inc_36 gs_mars_earth_scenario_inc_44 \
gs_mars_earth_scenario_inc_52 gs_mars_earth_scenario_inc_60 \
-s fcp lls lls_mip lls_lp

python3 main.py \
-e gs_mars_earth_scenario_inc_4 gs_mars_earth_scenario_inc_8 gs_mars_earth_scenario_inc_12 gs_mars_earth_scenario_inc_16 \
-s fcp random alternating lls lls_pat_unaware lls_mip lls_lp

python3 main.py \
-e gs_mars_earth_scenario_inc_4 gs_mars_earth_scenario_inc_8 gs_mars_earth_scenario_inc_12 gs_mars_earth_scenario_inc_16 \
gs_mars_earth_scenario_inc_20 gs_mars_earth_scenario_inc_24 gs_mars_earth_scenario_inc_28 gs_mars_earth_scenario_inc_32 \
-s fcp lls lls_mip lls_lp

python3 main.py \
-e gs_mars_earth_scenario_inc_4 gs_mars_earth_scenario_inc_8 gs_mars_earth_scenario_inc_12 gs_mars_earth_scenario_inc_16 \
-s fcp lls lls_mip lls_lp

python3 main.py \
-e gs_mars_earth_scenario_inc_4 gs_mars_earth_scenario_inc_8 gs_mars_earth_scenario_inc_12 gs_mars_earth_scenario_inc_16 \
gs_mars_earth_scenario_inc_20 gs_mars_earth_scenario_inc_24 gs_mars_earth_scenario_inc_28 gs_mars_earth_scenario_inc_32 \
gs_mars_earth_scenario_inc_36 gs_mars_earth_scenario_inc_40 \
-s fcp lls lls_mip lls_lp

python3 main.py \
-e gs_mars_earth_scenario_inc_44 gs_mars_earth_scenario_inc_48 \
gs_mars_earth_scenario_inc_52 gs_mars_earth_scenario_inc_56 gs_mars_earth_scenario_inc_60 gs_mars_earth_scenario_inc_64 \
-s fcp lls lls_lp && \
python3 main.py \
-e gs_mars_earth_scenario_inc_44 gs_mars_earth_scenario_inc_48 \
gs_mars_earth_scenario_inc_52 gs_mars_earth_scenario_inc_56 gs_mars_earth_scenario_inc_60 gs_mars_earth_scenario_inc_64 \
-s lls_mip