#!/bin/bash

uv run main.py \
    -e  gs_mars_earth_scenario_inc_4    \
    -e  gs_mars_earth_scenario_inc_12   \
    -e  gs_mars_earth_scenario_inc_20   \
    -e  gs_mars_earth_scenario_inc_28   \
    -e  gs_mars_earth_scenario_inc_36   \
    -e  gs_mars_earth_scenario_inc_44   \
    -e  gs_mars_earth_scenario_inc_52   \
    -e  gs_mars_earth_scenario_inc_60   \
    -s  fcp     \
    -s  lls     \
    -s  lls_mip \
    -s  lls_lp


uv run main.py \
    -e  gs_mars_earth_scenario_inc_4    \
    -e  gs_mars_earth_scenario_inc_8    \
    -e  gs_mars_earth_scenario_inc_12   \
    -e  gs_mars_earth_scenario_inc_16   \
    -s  fcp             \
    -s  random          \
    -s  alternating     \
    -s  lls             \
    -s  lls_pat_unaware \
    -s  lls_mip         \
    -s  lls_lp


uv run main.py \
    -e  gs_mars_earth_scenario_inc_4    \
    -e  gs_mars_earth_scenario_inc_8    \
    -e  gs_mars_earth_scenario_inc_12   \
    -e  gs_mars_earth_scenario_inc_16   \
    -e  gs_mars_earth_scenario_inc_20   \
    -e  gs_mars_earth_scenario_inc_24   \
    -e  gs_mars_earth_scenario_inc_28   \
    -e  gs_mars_earth_scenario_inc_32   \
    -s  fcp     \
    -s  lls     \
    -s  lls_mip \
    -s  lls_lp


uv run main.py \
    -e  gs_mars_earth_scenario_inc_4    \
    -e  gs_mars_earth_scenario_inc_8    \
    -e  gs_mars_earth_scenario_inc_12   \
    -e  gs_mars_earth_scenario_inc_16   \
    -s  fcp     \
    -s  lls     \
    -s  lls_mip \
    -s  lls_lp


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
    -s  fcp     \
    -s  lls     \
    -s  lls_mip \
    -s  lls_lp


uv run main.py \
    -e  gs_mars_earth_scenario_inc_44   \
    -e  gs_mars_earth_scenario_inc_48   \
    -e  gs_mars_earth_scenario_inc_52   \
    -e  gs_mars_earth_scenario_inc_56   \
    -e  gs_mars_earth_scenario_inc_60   \
    -e  gs_mars_earth_scenario_inc_64   \
    -s  fcp \
    -s  lls \
    -s  lls_lp && \
uv run main.py \
    -e  gs_mars_earth_scenario_inc_44   \
    -e  gs_mars_earth_scenario_inc_48   \
    -e  gs_mars_earth_scenario_inc_52   \
    -e  gs_mars_earth_scenario_inc_56   \
    -e  gs_mars_earth_scenario_inc_60   \
    -e  gs_mars_earth_scenario_inc_64   \
    -s  lls_mip


uv run main.py \
    -e  gs_mars_earth_scenario_inc_4    \
    -s  lls


uv run main.py \
    -e  gs_mars_earth_scenario_inc_reduced_4    \
    -e  gs_mars_earth_scenario_inc_reduced_8    \
    -e  gs_mars_earth_scenario_inc_reduced_12   \
    -e  gs_mars_earth_scenario_inc_reduced_16   \
    -e  gs_mars_earth_scenario_inc_reduced_20   \
    -s  fcp \
    -s  lls \
    -s  lls_mip