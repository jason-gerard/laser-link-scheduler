# laser-link-scheduler
Scheduling algorithm for free-space laser communication in deep space

## Usage
```
python3 lls.py -f sample_contact_plan
```

## ToDo
- [ ] Implement graph data structure to model contact plan. This could be a time-expanded graph, where the vertices are nodes and the edges are contacts, or a contact graph, where the vertices are contacts and the edges are the periods of time between contacts.
- [ ] Implement basic blossom algorithm that solves for a maximum matching
- [ ] Implement a weighted blossom algorithm that maximizes the matches based on throughput
- [ ] Implement some filtering logic on the contact plan to remove contacts that should not exist in the scenario i.e. Moon orbiter <> Earth relay contacts for a LunaNet deployment scenario. This should probably be done in the tool used to generate the contact plan, not here.