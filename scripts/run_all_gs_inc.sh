#!/bin/bash

uv run main.py \
    -e  gs_mars_earth_scenario_inc_reduced_4    \
        gs_mars_earth_scenario_inc_reduced_8    \
        gs_mars_earth_scenario_inc_reduced_12   \
        gs_mars_earth_scenario_inc_reduced_16   \
        gs_mars_earth_scenario_inc_reduced_20   \
        gs_mars_earth_scenario_inc_reduced_24   \
        gs_mars_earth_scenario_inc_reduced_28   \
        gs_mars_earth_scenario_inc_reduced_32   \
        gs_mars_earth_scenario_inc_reduced_36   \
        gs_mars_earth_scenario_inc_reduced_40   \
        gs_mars_earth_scenario_inc_reduced_44   \
        gs_mars_earth_scenario_inc_reduced_48   \
        gs_mars_earth_scenario_inc_reduced_52   \
        gs_mars_earth_scenario_inc_reduced_56   \
        gs_mars_earth_scenario_inc_reduced_60   \
        gs_mars_earth_scenario_inc_reduced_64   \
    -s  fcp random alternating lls lls_pat_unaware lls_mip