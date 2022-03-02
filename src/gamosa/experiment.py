from metrics_suite import MetricsSuite
import networkx as nx
from simulated_annealing import SimulatedAnnealing
from matplotlib import pyplot as plt
import numpy as np
from tests import *
import os
import random

def main2():

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
    sa = SimulatedAnnealing(filename, metrics_list=list(weights.keys()), weights=weights, cooling_schedule="linear_m", n_iters=1000, initial_config="grid", next_step="random_bounded")

    G2 = sa.anneal()
    

    fig1, (ax2, ax3) = plt.subplots(nrows=2, ncols=1)

    nx.draw(G, pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in G.nodes(data=True)]}, ax=ax2)
    
    nx.draw(G2, pos={k:np.array((v["x"], v["y"]),dtype=np.float32) for (k, v) in[u for u in G2.nodes(data=True)]}, ax=ax3)

    ms2 = MetricsSuite(G2, metrics_list=weights.keys())
    #ms2.calculate_metrics()
    #ms2.pretty_print_metrics()

    plt.show()

    sa.n_iters = 100
    sa.t_max = 1
    

    sa.plot_temperatures2()


def perform_sa(graph, metrics, initial, cooling, next_step, poly=0):

    ms = MetricsSuite(PATH + graph, metrics)
    #ms.write_graph("..\\..\\graph_drawings\\simulated_annealing\\EC-1_INITIAL-random_COOLING-linear_STEP-random-bounded\\__SA_" + graph)
    sa = SimulatedAnnealing(ms, initial, cooling, 0, 100, 0.6, next_step, 1, 1000, 60, 0, 0, n_polygon_sides=poly)

    G = sa.anneal()
    return G


def experiment_random_linear(filename, outfile, weights):
    G = perform_sa(filename, weights, "random", "linear_a", "random_bounded")
    ms_G = MetricsSuite(G)
    ms_G.write_graph(outfile)

def experiment_random_quadratic(filename, outfile, weights):
    G = perform_sa(filename, weights, "random", "quadratic_a", "random_bounded")
    ms_G = MetricsSuite(G)
    ms_G.write_graph(outfile)

def experiment_grid_linear(filename, outfile, weights):
    G = perform_sa(filename, weights, "grid", "linear_a", "random_bounded")
    ms_G = MetricsSuite(G)
    ms_G.write_graph(outfile)

def experiment_grid_quadratic(filename, outfile, weights):
    G = perform_sa(filename, weights, "grid", "quadratic_a", "random_bounded")
    ms_G = MetricsSuite(G)
    ms_G.write_graph(outfile)

def experiment_poly3_linear(filename, outfile, weights):
    G = perform_sa(filename, weights, "polygon", "linear_a", "random_bounded", poly=3)
    ms_G = MetricsSuite(G)
    ms_G.write_graph(outfile)

def experiment_poly3_quadratic(filename, outfile, weights):
    G = perform_sa(filename, weights, "polygon", "quadratic_a", "random_bounded", poly=3)
    ms_G = MetricsSuite(G)
    ms_G.write_graph(outfile)

def experiment_poly5_linear(filename, outfile, weights):
    G = perform_sa(filename, weights, "polygon", "linear_a", "random_bounded", poly=5)
    ms_G = MetricsSuite(G)
    ms_G.write_graph(outfile)

def experiment_poly5_quadratic(filename, outfile, weights):
    G = perform_sa(filename, weights, "polygon", "quadratic_a", "random_bounded", poly=5)
    ms_G = MetricsSuite(G)
    ms_G.write_graph(outfile)

def experiment_metric_loop(filename, i):
    metrics = {"edge_crossing": "EC",
                "edge_orthogonality": "EO",
                "angular_resolution": "AR",
                "edge_length": "EL",
                "gabriel_ratio": "GR",
                "crossing_angle": "CA",
    }

    for metric in metrics:
        # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-random_COOLING-linear_STEP-random-bounded\\"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-random_COOLING-linear_STEP-random-bounded\\SA_" + filename
        outfile = "..\\..\\graph_drawings\\simulated_annealing\\SA-" + str(i) + "_" + metrics[metric] + "-1_INITIAL-random_COOLING-linear_STEP-random-bounded_" + filename
        print(metric)
        experiment_random_linear(filename, outfile, {metric: 1})

    for metric in metrics:
        # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-random_COOLING-quadratic_STEP-random-bounded\\"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-random_COOLING-quadratic_STEP-random-bounded\\SA_" + filename
        outfile = "..\\..\\graph_drawings\\simulated_annealing\\SA-" + str(i) + "_" + metrics[metric] + "-1_INITIAL-random_COOLING-quadratic_STEP-random-bounded_" + filename
        print(metric)
        experiment_random_quadratic(filename, outfile, {metric: 1})

    for metric in metrics:
        # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-grid_COOLING-linear_STEP-random-bounded\\"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-grid_COOLING-linear_STEP-random-bounded\\SA_" + filename
        outfile = "..\\..\\graph_drawings\\simulated_annealing\\SA-" + str(i) + "_" + metrics[metric] + "-1_INITIAL-grid_COOLING-linear_STEP-random-bounded_" + filename
        print(metric)
        experiment_grid_linear(filename, outfile, {metric: 1})

    for metric in metrics:
        # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-grid_COOLING-quadratic_STEP-random-bounded\\"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-grid_COOLING-quadratic_STEP-random-bounded\\SA_" + filename
        outfile = "..\\..\\graph_drawings\\simulated_annealing\\SA-" + str(i) + "_" + metrics[metric] + "-1_INITIAL-grid_COOLING-quadratic_STEP-random-bounded_" + filename
        print(metric)
        experiment_grid_quadratic(filename, outfile, {metric: 1})

    for metric in metrics:
        # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly3_COOLING-linear_STEP-random-bounded\\"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly3_COOLING-linear_STEP-random-bounded\\SA_" + filename
        outfile = "..\\..\\graph_drawings\\simulated_annealing\\SA-" + str(i) + "_" + metrics[metric] + "-1_INITIAL-poly3_COOLING-linear_STEP-random-bounded_" + filename
        print(metric)
        experiment_poly3_linear(filename, outfile, {metric: 1})

    for metric in metrics:
        # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly3_COOLING-quadratic_STEP-random-bounded\\"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly3_COOLING-quadratic_STEP-random-bounded\\SA_" + filename
        outfile = "..\\..\\graph_drawings\\simulated_annealing\\SA-" + str(i) + "_" + metrics[metric] + "-1_INITIAL-poly3_COOLING-quadratic_STEP-random-bounded_" + filename
        print(metric)
        experiment_poly3_quadratic(filename, outfile, {metric: 1})

    for metric in metrics:
        # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly5_COOLING-linear_STEP-random-bounded\\"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly5_COOLING-linear_STEP-random-bounded\\SA_" + filename
        outfile = "..\\..\\graph_drawings\\simulated_annealing\\SA-" + str(i) + "_" + metrics[metric] + "-1_INITIAL-poly5_COOLING-linear_STEP-random-bounded_" + filename
        print(metric)
        experiment_poly5_linear(filename, outfile, {metric: 1})

    for metric in metrics:
        # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly5_COOLING-quadratic_STEP-random-bounded\\"
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly5_COOLING-quadratic_STEP-random-bounded\\SA_" + filename
        outfile = "..\\..\\graph_drawings\\simulated_annealing\\SA-" + str(i) + "_" + metrics[metric] + "-1_INITIAL-poly5_COOLING-quadratic_STEP-random-bounded_" + filename
        print(metric)
        experiment_poly5_quadratic(filename, outfile, {metric: 1})

def main():
    
    # Get file names
    directory = os.fsencode(PATH)
    fnames = []
    for file in os.listdir(directory):
        fnames.append(os.fsdecode(file))



    # Select random sample
    graphs = random.sample(fnames, 10)
    i = 1
    for graph in graphs:
        print(graph)
        experiment_metric_loop(graph, i)
        i += 1
        
    

    

if __name__ == "__main__":
    PATH = "..\\..\\graphs\\rome\\"
    main()