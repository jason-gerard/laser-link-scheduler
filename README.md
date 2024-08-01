# laser-link-scheduler
Scheduling algorithm for free-space laser communication in deep space

## Usage
```
python3 schedule.py -f mars_earth_simple_scenario
```

## Runs
| Version | Scenario                   | Duration        |
|---------|----------------------------|-----------------|
| V1      | mars_earth_simple_scenario | 4199.48 seconds |
| V1      | mars_earth_scenario        | est. 20 days    |
| V2      | mars_earth_simple_scenario | 1.2 seconds     |
| V2      | mars_earth_scenario        | 300 seconds     |

## ToDo
- [ ] Implement time expanded graph splitting
- [ ] Convert DS to numpy array based, this will make implementing the LP version simpler in the future
- [ ] Implement the fair contact plan algorithm
- [ ] Implement the random edge algorithm
- [ ] Create verification script to make sure all k states are properly matched
- [ ] Create some tests to make sure there are no regressions when refactoring
