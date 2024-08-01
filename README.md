# laser-link-scheduler
Scheduling algorithm for free-space laser communication in deep space

## Usage
```
python3 scheduler.py -f mars_earth_simple_scenario
```

## Testing
To run the tests
```
pytest -vv -s
```

To update the static mocks for regression testing
```
python3 update_static_mocks.py
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
- [ ] Implement the fair contact plan algorithm
- [ ] Implement the random edge algorithm
- [ ] Create some tests to make sure there are no regressions when refactoring
