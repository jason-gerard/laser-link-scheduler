#!/bin/bash

uv run main.py \
    -e  dte_mars_earth_scenario_inc_8  \
    -e  dte_mars_earth_scenario_inc_16  \
    -e  dte_mars_earth_scenario_inc_20  \
    -e  dte_mars_earth_scenario_inc_24  \
    -e  dte_mars_earth_scenario_inc_28  \
    -e  dte_mars_earth_scenario_inc_32  \
    -e  dte_mars_earth_scenario_inc_36  \
    -e  dte_mars_earth_scenario_inc_40  \
    -e  dte_mars_earth_scenario_inc_44  \
    -e  dte_mars_earth_scenario_inc_48  \
    -e  dte_mars_earth_scenario_inc_52  \
    -e  dte_mars_earth_scenario_inc_56  \
    -e  dte_mars_earth_scenario_inc_60  \
    -e  dte_mars_earth_scenario_inc_64  \
    -s  fcp             \
    -s  lls             \
    -s  lls_pat_unaware \
    -s  energy_aware    \
    -s  battery_energy  \
    -s  lifespan_aware
    # -s  lls_mip         \
    # -s  lls_lp


uv run main.py \
    -e  gs_mars_earth_scenario_4  \
    -e  gs_mars_earth_scenario_8  \
    -e  gs_mars_earth_scenario_16  \
    -e  gs_mars_earth_scenario_20  \
    -e  gs_mars_earth_scenario_24  \
    -s  fcp             \
    -s  lls             \
    -s  lls_pat_unaware \
    -s  energy_aware    \
    -s  battery_energy  \
    -s  lifespan_aware
    # -s  lls_mip         \
    # -s  lls_lp

uv run main.py \
    -e  gs_mars_earth_scenario_inc_4  \
    -e  gs_mars_earth_scenario_inc_8  \
    -e  gs_mars_earth_scenario_inc_16  \
    -e  gs_mars_earth_scenario_inc_20  \
    -e  gs_mars_earth_scenario_inc_24  \
    -e  gs_mars_earth_scenario_inc_28  \
    -e  gs_mars_earth_scenario_inc_32  \
    -e  gs_mars_earth_scenario_inc_36  \
    -e  gs_mars_earth_scenario_inc_40  \
    -e  gs_mars_earth_scenario_inc_44  \
    -e  gs_mars_earth_scenario_inc_48  \
    -e  gs_mars_earth_scenario_inc_52  \
    -e  gs_mars_earth_scenario_inc_56  \
    -e  gs_mars_earth_scenario_inc_60  \
    -e  gs_mars_earth_scenario_inc_64  \
    -s  fcp             \
    -s  lls             \
    -s  lls_pat_unaware \
    -s  energy_aware    \
    -s  battery_energy  \
    -s  lifespan_aware
    # -s  lls_mip         \
    # -s  lls_lp


uv run main.py \
    -e  gs_mars_earth_scenario_inc_reduced_4  \
    -e  gs_mars_earth_scenario_inc_reduced_8  \
    -e  gs_mars_earth_scenario_inc_reduced_16  \
    -e  gs_mars_earth_scenario_inc_reduced_20  \
    -e  gs_mars_earth_scenario_inc_reduced_24  \
    -e  gs_mars_earth_scenario_inc_reduced_28  \
    -e  gs_mars_earth_scenario_inc_reduced_32  \
    -e  gs_mars_earth_scenario_inc_reduced_36  \
    -e  gs_mars_earth_scenario_inc_reduced_40  \
    -e  gs_mars_earth_scenario_inc_reduced_44  \
    -e  gs_mars_earth_scenario_inc_reduced_48  \
    -e  gs_mars_earth_scenario_inc_reduced_52  \
    -e  gs_mars_earth_scenario_inc_reduced_56  \
    -e  gs_mars_earth_scenario_inc_reduced_60  \
    -e  gs_mars_earth_scenario_inc_reduced_64  \
    -s  fcp             \
    -s  lls             \
    -s  lls_pat_unaware \
    -s  energy_aware    \
    -s  battery_energy  \
    -s  lifespan_aware
    # -s  lls_mip         \
    # -s  lls_lp

