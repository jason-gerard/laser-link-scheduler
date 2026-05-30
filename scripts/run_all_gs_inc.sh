#!/bin/bash

uv run main.py \
    -e  gs_mars_earth_scenario_inc_4    \
    -e  gs_mars_earth_scenario_inc_8    \
    -e  gs_mars_earth_scenario_inc_12   \
    -e  gs_mars_earth_scenario_inc_16   \
    -e  gs_mars_earth_scenario_inc_20   \
    -e  gs_mars_earth_scenario_inc_24   \
    -e  gs_mars_earth_scenario_inc_28   \
    -e  gs_mars_earth_scenario_inc_32   \
    -e  gs_mars_earth_scenario_inc_36   \
    -e  gs_mars_earth_scenario_inc_40   \
    -e  gs_mars_earth_scenario_inc_44   \
    -e  gs_mars_earth_scenario_inc_48   \
    -e  gs_mars_earth_scenario_inc_52   \
    -e  gs_mars_earth_scenario_inc_56   \
    -e  gs_mars_earth_scenario_inc_60   \
    -e  gs_mars_earth_scenario_inc_64   \
    -s  lls             \
    -s  fcp             \
    -s  energy_aware    \
    -s  battery_energy  \
    -s  lifespan_aware  \