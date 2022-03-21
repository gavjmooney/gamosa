from metrics_suite import MetricsSuite
import networkx as nx
from simulated_annealing import SimulatedAnnealing
from matplotlib import pyplot as plt
import numpy as np
from tests import *



def main():

    filename = "..\\..\\graphs\\moon\\poly.graphml"
    filename = "..\\..\\graph_drawings\\nathan\\B_BB_FR_ABC_fr_10_data_110_BB.graphml"

    ms = MetricsSuite(filename)
    #ms.calculate_metric("angular_resolution")
    #ms.pretty_print_metrics()
    G = ms.graph

    weights = { "crossing_angle":1}   
    #weights = { "angular_resolution":1, "edge_length":1}   
    #weights = { "gabriel_ratio":1, "node_resolution":1}   
    #weights = {"edge_length":1, "edge_crossing":2, "node_resolution":1, "angular_resolution":1, "gabriel_ratio":1}
    # weights = { "edge_crossing":5, "node_resolution":1, "angular_resolution":1, "edge_length":1 }
    #weights = { "node_resolution":1, "edge_length":1 }
    ms = MetricsSuite(filename, weights)
    sa = SimulatedAnnealing(ms, cooling_schedule="linear_a", n_iters=1000, initial_config="polygon",next_step="random_bounded", n_polygon_sides=5)

    ms.draw_graph(sa.initial_config)
    ms.write_graph("polyout.graphml", sa.initial_config)
    #ms.calculate_metrics()
    #ms.pretty_print_metrics()
    #G2 = sa.anneal()
    

    #fig1, (ax2, ax3) = plt.subplots(nrows=2, ncols=1)

    #ms.draw_graph(sa.initial_config)
    #ms.draw_graph(G2)

    #nx.draw(sa.initial_config, pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in sa.initial_config.nodes(data=True)]}, ax=ax2)
    
    #nx.draw(G2, pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in G2.nodes(data=True)]}, ax=ax3)

    #ms2 = MetricsSuite(G2, weights)
    #ms2.calculate_metrics()
    #ms2.pretty_print_metrics()

    plt.show()

    # sa.n_iters = 100
    # sa.t_max = 1
    

    # sa.plot_temperatures2()

    

if __name__ == "__main__":
    main()