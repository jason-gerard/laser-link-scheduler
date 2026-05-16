#!/bin/bash

uv run main.py \
    -e  dte_mars_earth_scenario_inc_20  \
    -e  dte_mars_earth_scenario_inc_16  \
    -e  dte_mars_earth_scenario_inc_24  \
    -e  dte_mars_earth_scenario_inc_28  \
    -e  dte_mars_earth_scenario_inc_32  \
    -e  dte_mars_earth_scenario_inc_36  \
    -e  dte_mars_earth_scenario_inc_40  \
    -e  dte_mars_earth_scenario_inc_44  \
    -s  fcp            \
    -s  energy_aware   \
    -s  battery_energy \
    -s  lifespan_aware