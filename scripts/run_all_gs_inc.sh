#!/bin/bash

uv run main.py \
    -e  gs_mars_earth_scenario_inc_reduced_4    \
    -e  gs_mars_earth_scenario_inc_reduced_8    \
    -e  gs_mars_earth_scenario_inc_reduced_12   \
    -e  gs_mars_earth_scenario_inc_reduced_16   \
    -e  gs_mars_earth_scenario_inc_reduced_20   \
    -e  gs_mars_earth_scenario_inc_reduced_24   \
    -e  gs_mars_earth_scenario_inc_reduced_28   \
    -e  gs_mars_earth_scenario_inc_reduced_32   \
    -e  gs_mars_earth_scenario_inc_reduced_36   \
    -e  gs_mars_earth_scenario_inc_reduced_40   \
    -e  gs_mars_earth_scenario_inc_reduced_44   \
    -e  gs_mars_earth_scenario_inc_reduced_48   \
    -e  gs_mars_earth_scenario_inc_reduced_52   \
    -e  gs_mars_earth_scenario_inc_reduced_56   \
    -e  gs_mars_earth_scenario_inc_reduced_60   \
    -e  gs_mars_earth_scenario_inc_reduced_64   \
    -s  fcp             \
    -s  random          \
    -s  alternating     \
    -s  lls             \
    -s  lls_pat_unaware \
    -s  lls_mip