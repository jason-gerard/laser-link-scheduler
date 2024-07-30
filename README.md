# laser-link-scheduler
Scheduling algorithm for free-space laser communication in deep space

## Usage
```
python3 lls.py -f mars_earth_simple_scenario
```

## Assumption
A key assumption we make is that contacts are bidirectional i.e. if there is a contact from A -> B
then there is also a contact from B -> A. These contacts can also exist at the same time in the contact plan. Since we are dealing
with lasers this might not be a valid assumption but since our data is unidirectional i.e. we are only
transmitting data from Mars -> Earth, it doesn't matter anyway. This assumption should be revisited in the future
as the scenarios become more complex. This data is embedded in the contact plan and not the code, so the code should
already be written to support unidirectional contacts but logically, for now, all contacts are bidirectional. This would pose a problem in the future because most of the matching algorithms assume an undirected graph.

## ToDo
- Convert arrays to numpy
- Update methods to give weights matrix
- Implement cpp Blossom algorithm with python bindings
- Update DS to use DS from paper i.e. W, L, T, etc

## Notes
- Note on topology optimization: If we restrict orbiter -> orbiter contacts then we don't even need the blossom algorithm, this might actually perform better while not sacrificing anything for our use case. this would only contain orbiter -> relay and relay -> relay contacts.
