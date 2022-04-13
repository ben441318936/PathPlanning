from Grid import Grid
from Agent import Agent

if __name__ == "__main__":
    G = Grid((10,10))
    G.place_agent(3,3)
    G.place_target(4,4)
    A = Agent((10,10))
    
    A.print_map()

    A.update_map(G.scan(A.cone_of_vision()))

    A.print_map()


