#!/bin/bash

uv run main.py \
    -e mars_relay_earth_scenario_inc_4 \
    -e mars_relay_earth_scenario_inc_8 \
    -e mars_relay_earth_scenario_inc_12 \
    -e mars_relay_earth_scenario_inc_16 \
    -e mars_relay_earth_scenario_inc_20 \
    -e mars_relay_earth_scenario_inc_24 \
    -e mars_relay_earth_scenario_inc_28 \
    -e mars_relay_earth_scenario_inc_32 \
    -e mars_relay_earth_scenario_inc_36 \
    -e mars_relay_earth_scenario_inc_40 \
    -e mars_relay_earth_scenario_inc_44 \
    -e mars_relay_earth_scenario_inc_48 \
    -e mars_relay_earth_scenario_inc_52 \
    -e mars_relay_earth_scenario_inc_56 \
    -e mars_relay_earth_scenario_inc_60 \
    -e mars_relay_earth_scenario_inc_64 \
    -s  lls            \
    -s  fcp            \
    -s  energy_aware   \
    -s  battery_energy \
    -s  lifespan_aware