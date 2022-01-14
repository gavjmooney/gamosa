from metrics_suite import MetricsSuite
import networkx as nx
from simulated_annealing import SimulatedAnnealing
from matplotlib import pyplot as plt
import numpy as np
from tests import *



def main():

    filename = "..\\..\\graphs\\moon\\ca.graphml"

    ms = MetricsSuite(filename)
    #ms.calculate_metric("angular_resolution")
    #ms.pretty_print_metrics()
    G = ms.graph

    weights = { "edge_crossing":1}   
    #weights = { "angular_resolution":1, "edge_length":1}   
    #weights = { "gabriel_ratio":1, "node_resolution":1}   
    #weights = {"edge_length":1, "edge_crossing":2, "node_resolution":1, "angular_resolution":1, "gabriel_ratio":1}
    # weights = { "edge_crossing":5, "node_resolution":1, "angular_resolution":1, "edge_length":1 }
    #weights = { "node_resolution":1, "edge_length":1 }
    sa = SimulatedAnnealing(filename, metrics_list=list(weights.keys()), weights=weights, cooling_schedule="linear_m", n_iters=1000, initial_config="load")

    G2 = sa.anneal()
    

    fig1, (ax2, ax3) = plt.subplots(nrows=2, ncols=1)

    nx.draw(G, pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in G.nodes(data=True)]}, ax=ax2)
    
    nx.draw(G2, pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in G2.nodes(data=True)]}, ax=ax3)

    ms2 = MetricsSuite(G2, metrics_list=weights.keys())
    ms2.calculate_metrics()
    ms2.pretty_print_metrics()

    plt.show()

if __name__ == "__main__":
    main()