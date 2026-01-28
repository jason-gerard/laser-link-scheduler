#!/bin/bash

uv run main.py \
    -e  gs_mars_earth_xs_scenario   \
        gs_mars_earth_s_scenario    \
        gs_mars_earth_m_scenario    \
        gs_mars_earth_l_scenario    \
        gs_mars_earth_xl_scenario   \
    -s  fcp random alternating lls lls_pat_unaware lls_mip lls_lp