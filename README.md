# laser-link-scheduler
Scheduling algorithm for free-space laser communication in deep space

## Usage
There are a couple scenarios the project comes with by default. These are the ones used for analysis. They model interplanetary Mars to Earth space exploration missions where there are a set of orbiters around Mars that either produce data with onboard equipment or receive data from other nodes such as landers, rovers, or drones. The orbiters transmit data to relay satellites around Mars which then try to transmit data across interplanetary distances to Earth.

You can run any of the scenarios by passing in the name of the experiment and algorithm.
```
python3 main.py -e <experiment_name> -s <scheduler_name>
```
For example running the following command will run the `mars_earth_simple_scenario` with the `Laser Link Scheduler` algorithm
```
python3 main.py -e mars_earth_simple_scenario -f lls
```

Get the list of all input parameters
```
python3 main.py --help
```

## Related Tools
- Orbit-generator, https://github.com/jason-gerard/orbit-generator, can be used to define base orbits for constellations on different planets.
- IPN-D and IPN-V, https://gitlab.inria.fr/jfraire/ipn-v, can be used to generate the full orbits for each node (using a 2 body propagator), generate a LoS based contact plan, and visualize the contact plan.
- This tool outputs the scheduled contact plan in the standard ION format which can be used in discrete event Monte Carlo simulators such as DtnSim, https://gitlab.inria.fr/jfraire/dtnsim, to actually evaluate how different routing algorithms, such as contact graph routing (CGR), perform on the scheduled contact plan.

## Testing
To run the tests
```
pytest -vv -s
```

## Runs
| Version | Scenario                   | Duration        |
|---------|----------------------------|-----------------|
| V1      | mars_earth_simple_scenario | 4199.48 seconds |
| V1      | mars_earth_scenario        | est. 20 days    |
| V2      | mars_earth_simple_scenario | 1.2 seconds     |
| V2      | mars_earth_scenario        | 305.7 seconds   |

## ToDo
- [ ] Implement time expanded graph splitting
- [ ] Implement the fair contact plan algorithm
- [ ] Implement the random edge algorithm
- [ ] Write exporter back to IPN-V contact plan format
- [ ] Write reporting output file to compare scheduling algorithms easier