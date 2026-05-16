#!/bin/bash

uv run main.py \
    -e  gs_mars_earth_xs_scenario   \
    -e  gs_mars_earth_s_scenario    \
    -e  gs_mars_earth_m_scenario    \
    -e  gs_mars_earth_l_scenario    \
    -e  gs_mars_earth_xl_scenario   \
    -s  fcp             \
    -s  random          \
    -s  alternating     \
    -s  lls             \
    -s  lls_pat_unaware \
    -s  lls_mip         \
    -s  lls_lp          \
    -s  energy_aware    \
    -s  battery_energy  \
    -s  lifespan_aware