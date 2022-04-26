# A robot planning and controls project


### Grid.py
Implements a grid-like environment that supports 8-connected grid movement. Also supports "scanning" of grid status in a given area (for robot perception).

### Agent.py
Implements naive A* and D* Lite for goal-direction navigation in unknown environment. The agent plans with the assumption that unobserved cells in the grid are empty. Then replans as new cells are observed or as the map is expanded. The naive A* agent will fully replan everytime the map changes. The D* Lite agent will replan using previous estimates. Also implements an unified Agent interface shared by the agents who use different planning algorithms.

### Simulation.py
Implements a simulation loop using Grid and Agent interfaces. Also implements visualizations for the Grid environment and Agent's internal map using PyGames.