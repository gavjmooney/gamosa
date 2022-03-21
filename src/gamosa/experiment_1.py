from fileinput import filename
from metrics_suite import MetricsSuite
import networkx as nx
from simulated_annealing import SimulatedAnnealing
from matplotlib import pyplot as plt
import numpy as np
from tests import *
import os
import random
import shutil

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


def perform_sa(graph, metrics, initial, cooling, next_step, outfile_i, poly=0):

    ms = MetricsSuite(PATH + graph, metrics)
    #ms.write_graph("..\\..\\graph_drawings\\simulated_annealing\\EC-1_INITIAL-random_COOLING-linear\\__SA_" + graph)
    sa = SimulatedAnnealing(ms, initial, cooling, 0, 100, 0.6, next_step, 1, 1000, 60, 0, 0, n_polygon_sides=poly)
    ms.write_graph(outfile_i, sa.initial_config)
    G = sa.anneal()
    return {"i":sa.initial_config, "f":G}


def experiment_random_linear(filename, outfile_i, outfile_f, stat_file, weights, metric):
    GS = perform_sa(filename, weights, "random", "linear_a", "random_bounded", outfile_i)
    ms_i = MetricsSuite(GS["i"], {metric:1})
    ms_i2 = MetricsSuite(GS["i"], all_metrics)
    ms_G = MetricsSuite(GS["f"], {metric:1})
    ms_G2 = MetricsSuite(GS["f"], all_metrics)
    ms_G.write_graph(outfile_f)

    metric_list = [str(m) for m in get_metric_weights(metric)]
    line_list = [outfile_f[63:]]
    line_list.extend(metric_list)
    line_list.extend(["random", "linear"])
    a = ms_i.combine_metrics()
    a2 = ms_i2.combine_metrics()
    b = ms_G.combine_metrics()
    c = ms_G2.combine_metrics()
    if a:
        a = str(round(a,3))
    else:
        a = "None"
    if a2:
        a2 = str(round(a2,3))
    else:
        a2 = "None"
    if b:
        b = str(round(b,3))
    else:
        b = "None"
    if c:
        c = str(round(c,3))
    else:
        c = "None"
    line_list.append(a)
    line_list.append(a2)
    line_list.append(b)
    line_list.append(c)
    if metric == "crossing_angle":
        line_list.append(str(ms_i.metrics["edge_crossing"]["num_crossings"]))
        line_list.append(str(ms_G.metrics["edge_crossing"]["num_crossings"]))
    else:
        line_list.append("N/A")
        line_list.append("N/A")
    line = ",".join(line_list) + "\n"
    stat_file.write(line)

def experiment_random_quadratic(filename, outfile_i, outfile_f, stat_file, weights, metric):
    GS = perform_sa(filename, weights, "random", "quadratic_a", "random_bounded", outfile_i)
    ms_i = MetricsSuite(GS["i"], {metric:1})
    ms_i2 = MetricsSuite(GS["i"], all_metrics)
    ms_G = MetricsSuite(GS["f"], {metric:1})
    ms_G2 = MetricsSuite(GS["f"], all_metrics)
    ms_G.write_graph(outfile_f)

    metric_list = [str(m) for m in get_metric_weights(metric)]
    line_list = [outfile_f[63:]]
    line_list.extend(metric_list)
    line_list.extend(["random", "quadratic"])
    a = ms_i.combine_metrics()
    a2 = ms_i2.combine_metrics()
    b = ms_G.combine_metrics()
    c = ms_G2.combine_metrics()
    if a:
        a = str(round(a,3))
    else:
        a = "None"
    if a2:
        a2 = str(round(a2,3))
    else:
        a2 = "None"
    if b:
        b = str(round(b,3))
    else:
        b = "None"
    if c:
        c = str(round(c,3))
    else:
        c = "None"
    line_list.append(a)
    line_list.append(a2)
    line_list.append(b)
    line_list.append(c)
    if metric == "crossing_angle":
        line_list.append(str(ms_i.metrics["edge_crossing"]["num_crossings"]))
        line_list.append(str(ms_G.metrics["edge_crossing"]["num_crossings"]))
    else:
        line_list.append("N/A")
        line_list.append("N/A")
    line = ",".join(line_list) + "\n"
    stat_file.write(line)

def experiment_grid_linear(filename, outfile_i, outfile_f, stat_file, weights, metric):
    GS = perform_sa(filename, weights, "grid", "linear_a", "random_bounded", outfile_i)
    ms_i = MetricsSuite(GS["i"], {metric:1})
    ms_i2 = MetricsSuite(GS["i"], all_metrics)
    ms_G = MetricsSuite(GS["f"], {metric:1})
    ms_G2 = MetricsSuite(GS["f"], all_metrics)
    ms_G.write_graph(outfile_f)

    metric_list = [str(m) for m in get_metric_weights(metric)]
    line_list = [outfile_f[63:]]
    line_list.extend(metric_list)
    line_list.extend(["grid", "linear"])
    a = ms_i.combine_metrics()
    a2 = ms_i2.combine_metrics()
    b = ms_G.combine_metrics()
    c = ms_G2.combine_metrics()
    if a:
        a = str(round(a,3))
    else:
        a = "None"
    if a2:
        a2 = str(round(a2,3))
    else:
        a2 = "None"
    if b:
        b = str(round(b,3))
    else:
        b = "None"
    if c:
        c = str(round(c,3))
    else:
        c = "None"
    line_list.append(a)
    line_list.append(a2)
    line_list.append(b)
    line_list.append(c)
    if metric == "crossing_angle":
        line_list.append(str(ms_i.metrics["edge_crossing"]["num_crossings"]))
        line_list.append(str(ms_G.metrics["edge_crossing"]["num_crossings"]))
    else:
        line_list.append("N/A")
        line_list.append("N/A")
    line = ",".join(line_list) + "\n"
    stat_file.write(line)

def experiment_grid_quadratic(filename, outfile_i, outfile_f, stat_file, weights, metric):
    GS = perform_sa(filename, weights, "grid", "quadratic_a", "random_bounded", outfile_i)
    ms_i = MetricsSuite(GS["i"], {metric:1})
    ms_i2 = MetricsSuite(GS["i"], all_metrics)
    ms_G = MetricsSuite(GS["f"], {metric:1})
    ms_G2 = MetricsSuite(GS["f"], all_metrics)
    ms_G.write_graph(outfile_f)

    metric_list = [str(m) for m in get_metric_weights(metric)]
    line_list = [outfile_f[63:]]
    line_list.extend(metric_list)
    line_list.extend(["grid", "quadratic"])
    a = ms_i.combine_metrics()
    a2 = ms_i2.combine_metrics()
    b = ms_G.combine_metrics()
    c = ms_G2.combine_metrics()
    if a:
        a = str(round(a,3))
    else:
        a = "None"
    if a2:
        a2 = str(round(a2,3))
    else:
        a2 = "None"
    if b:
        b = str(round(b,3))
    else:
        b = "None"
    if c:
        c = str(round(c,3))
    else:
        c = "None"
    line_list.append(a)
    line_list.append(a2)
    line_list.append(b)
    line_list.append(c)
    if metric == "crossing_angle":
        line_list.append(str(ms_i.metrics["edge_crossing"]["num_crossings"]))
        line_list.append(str(ms_G.metrics["edge_crossing"]["num_crossings"]))
    else:
        line_list.append("N/A")
        line_list.append("N/A")
    line = ",".join(line_list) + "\n"
    stat_file.write(line)

def experiment_poly3_linear(filename, outfile_i, outfile_f, stat_file, weights, metric):
    GS = perform_sa(filename, weights, "polygon", "linear_a", "random_bounded", outfile_i, poly=3)
    ms_i = MetricsSuite(GS["i"], {metric:1})
    ms_i2 = MetricsSuite(GS["i"], all_metrics)
    ms_G = MetricsSuite(GS["f"], {metric:1})
    ms_G2 = MetricsSuite(GS["f"], all_metrics)
    ms_G.write_graph(outfile_f)

    metric_list = [str(m) for m in get_metric_weights(metric)]
    line_list = [outfile_f[63:]]
    line_list.extend(metric_list)
    line_list.extend(["poly3", "linear"])
    a = ms_i.combine_metrics()
    a2 = ms_i2.combine_metrics()
    b = ms_G.combine_metrics()
    c = ms_G2.combine_metrics()
    if a:
        a = str(round(a,3))
    else:
        a = "None"
    if a2:
        a2 = str(round(a2,3))
    else:
        a2 = "None"
    if b:
        b = str(round(b,3))
    else:
        b = "None"
    if c:
        c = str(round(c,3))
    else:
        c = "None"
    line_list.append(a)
    line_list.append(a2)
    line_list.append(b)
    line_list.append(c)
    if metric == "crossing_angle":
        line_list.append(str(ms_i.metrics["edge_crossing"]["num_crossings"]))
        line_list.append(str(ms_G.metrics["edge_crossing"]["num_crossings"]))
    else:
        line_list.append("N/A")
        line_list.append("N/A")
    line = ",".join(line_list) + "\n"
    stat_file.write(line)

def experiment_poly3_quadratic(filename, outfile_i, outfile_f, stat_file, weights, metric):
    GS = perform_sa(filename, weights, "polygon", "quadratic_a", "random_bounded", outfile_i, poly=3)
    ms_i = MetricsSuite(GS["i"], {metric:1})
    ms_i2 = MetricsSuite(GS["i"], all_metrics)
    ms_G = MetricsSuite(GS["f"], {metric:1})
    ms_G2 = MetricsSuite(GS["f"], all_metrics)
    ms_G.write_graph(outfile_f)

    metric_list = [str(m) for m in get_metric_weights(metric)]
    line_list = [outfile_f[63:]]
    line_list.extend(metric_list)
    line_list.extend(["poly3", "quadratic"])
    a = ms_i.combine_metrics()
    a2 = ms_i2.combine_metrics()
    b = ms_G.combine_metrics()
    c = ms_G2.combine_metrics()
    if a:
        a = str(round(a,3))
    else:
        a = "None"
    if a2:
        a2 = str(round(a2,3))
    else:
        a2 = "None"
    if b:
        b = str(round(b,3))
    else:
        b = "None"
    if c:
        c = str(round(c,3))
    else:
        c = "None"
    line_list.append(a)
    line_list.append(a2)
    line_list.append(b)
    line_list.append(c)
    if metric == "crossing_angle":
        line_list.append(str(ms_i.metrics["edge_crossing"]["num_crossings"]))
        line_list.append(str(ms_G.metrics["edge_crossing"]["num_crossings"]))
    else:
        line_list.append("N/A")
        line_list.append("N/A")
    line = ",".join(line_list) + "\n"
    stat_file.write(line)

def experiment_poly5_linear(filename, outfile_i, outfile_f, stat_file, weights, metric):
    GS = perform_sa(filename, weights, "polygon", "linear_a", "random_bounded", outfile_i, poly=5)
    ms_i = MetricsSuite(GS["i"], {metric:1})
    ms_i2 = MetricsSuite(GS["i"], all_metrics)
    ms_G = MetricsSuite(GS["f"], {metric:1})
    ms_G2 = MetricsSuite(GS["f"], all_metrics)
    ms_G.write_graph(outfile_f)

    metric_list = [str(m) for m in get_metric_weights(metric)]
    line_list = [outfile_f[63:]]
    line_list.extend(metric_list)
    line_list.extend(["poly5", "linear"])
    a = ms_i.combine_metrics()
    a2 = ms_i2.combine_metrics()
    b = ms_G.combine_metrics()
    c = ms_G2.combine_metrics()
    if a:
        a = str(round(a,3))
    else:
        a = "None"
    if a2:
        a2 = str(round(a2,3))
    else:
        a2 = "None"
    if b:
        b = str(round(b,3))
    else:
        b = "None"
    if c:
        c = str(round(c,3))
    else:
        c = "None"
    line_list.append(a)
    line_list.append(a2)
    line_list.append(b)
    line_list.append(c)
    if metric == "crossing_angle":
        line_list.append(str(ms_i.metrics["edge_crossing"]["num_crossings"]))
        line_list.append(str(ms_G.metrics["edge_crossing"]["num_crossings"]))
    else:
        line_list.append("N/A")
        line_list.append("N/A")
    line = ",".join(line_list) + "\n"
    stat_file.write(line)

def experiment_poly5_quadratic(filename, outfile_i, outfile_f, stat_file, weights, metric):
    GS = perform_sa(filename, weights, "polygon", "quadratic_a", "random_bounded", outfile_i, poly=5)
    ms_i = MetricsSuite(GS["i"], {metric:1})
    ms_i2 = MetricsSuite(GS["i"], all_metrics)
    ms_G = MetricsSuite(GS["f"], {metric:1})
    ms_G2 = MetricsSuite(GS["f"], all_metrics)
    ms_G.write_graph(outfile_f)

    metric_list = [str(m) for m in get_metric_weights(metric)]
    line_list = [outfile_f[63:]]
    line_list.extend(metric_list)
    line_list.extend(["poly5", "quadratic"])
    a = ms_i.combine_metrics()
    a2 = ms_i2.combine_metrics()
    b = ms_G.combine_metrics()
    c = ms_G2.combine_metrics()
    if a:
        a = str(round(a,3))
    else:
        a = "None"
    if a2:
        a2 = str(round(a2,3))
    else:
        a2 = "None"
    if b:
        b = str(round(b,3))
    else:
        b = "None"
    if c:
        c = str(round(c,3))
    else:
        c = "None"
    line_list.append(a)
    line_list.append(a2)
    line_list.append(b)
    line_list.append(c)
    if metric == "crossing_angle":
        line_list.append(str(ms_i.metrics["edge_crossing"]["num_crossings"]))
        line_list.append(str(ms_G.metrics["edge_crossing"]["num_crossings"]))
    else:
        line_list.append("N/A")
        line_list.append("N/A")
    line = ",".join(line_list) + "\n"
    stat_file.write(line)


def get_metric_weights(metric):
    if metric == "edge_crossing":
        return [1,0,0,0,0,0]
    elif metric == "edge_orthogonality":
        return [0,1,0,0,0,0]
    elif metric == "angular_resolution":
        return [0,0,1,0,0,0]
    elif metric == "edge_length":
        return [0,0,0,1,0,0]
    elif metric == "gabriel_ratio":
        return [0,0,0,0,1,0]
    elif metric == "crossing_angle":
        return [0,0,0,0,0,1]


def experiment_metric_loop(filename, i):
    metrics = {"edge_crossing": "EC",
                "edge_orthogonality": "EO",
                "angular_resolution": "AR",
                "edge_length": "EL",
                "gabriel_ratio": "GR",
                "crossing_angle": "CA",
    }

    stat_file = "..\\..\\data\\experiment_1.csv"
    with open(stat_file, "a") as stat_f:

        # legend = "filename,EC,EO,AR,EL,GR,CA,Initial Config,Cooling Schedule,Initial Evaluation(Chosen Metrics),Final Evaluation(Chosen Metics),Final Evaluation(All Metrics)\n"
        # stat_f.write(legend)

        for metric in metrics:
            # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-random_COOLING-linear\\"
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-random_COOLING-linear\\SA"
            outfile_i = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "I" +"_" + metrics[metric] + "-1_INITIAL-random_COOLING-linear.graphml"
            outfile_f = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "F" +"_" + metrics[metric] + "-1_INITIAL-random_COOLING-linear.graphml"
            print(metric)
            experiment_random_linear(filename, outfile_i, outfile_f, stat_f, {metric: 1}, metric)

        for metric in metrics:
            # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-random_COOLING-quadratic\\"
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-random_COOLING-quadratic\\SA_"
            outfile_i = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "I" +"_" + metrics[metric] + "-1_INITIAL-random_COOLING-quadratic.graphml"
            outfile_f = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "F" +"_" + metrics[metric] + "-1_INITIAL-random_COOLING-quadratic.graphml"
            print(metric)
            experiment_random_quadratic(filename, outfile_i, outfile_f, stat_f, {metric: 1}, metric)

        for metric in metrics:
            # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-grid_COOLING-linear\\"
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-grid_COOLING-linear\\SA_"
            outfile_i = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "I" +"_" + metrics[metric] + "-1_INITIAL-grid_COOLING-linear.graphml"
            outfile_f = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "F" +"_" + metrics[metric] + "-1_INITIAL-grid_COOLING-linear.graphml"
            print(metric)
            experiment_grid_linear(filename, outfile_i, outfile_f, stat_f, {metric: 1}, metric)

        for metric in metrics:
            # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-grid_COOLING-quadratic\\"
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-grid_COOLING-quadratic\\SA_"
            outfile_i = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "I" +"_" + metrics[metric] + "-1_INITIAL-grid_COOLING-quadratic.graphml"
            outfile_f = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "F" +"_" + metrics[metric] + "-1_INITIAL-grid_COOLING-quadratic.graphml"
            print(metric)
            experiment_grid_quadratic(filename, outfile_i, outfile_f, stat_f, {metric: 1}, metric)

        for metric in metrics:
            # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly3_COOLING-linear\\"
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly3_COOLING-linear\\SA_"
            outfile_i = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "I" +"_" + metrics[metric] + "-1_INITIAL-poly3_COOLING-linear.graphml"
            outfile_f = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "F" +"_" + metrics[metric] + "-1_INITIAL-poly3_COOLING-linear.graphml"
            print(metric)
            experiment_poly3_linear(filename, outfile_i, outfile_f, stat_f, {metric: 1}, metric)

        for metric in metrics:
            # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly3_COOLING-quadratic\\"
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly3_COOLING-quadratic\\SA_"
            outfile_i = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "I" +"_" + metrics[metric] + "-1_INITIAL-poly3_COOLING-quadratic.graphml"
            outfile_f = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "F" +"_" + metrics[metric] + "-1_INITIAL-poly3_COOLING-quadratic.graphml"
            print(metric)
            experiment_poly3_quadratic(filename, outfile_i, outfile_f, stat_f, {metric: 1}, metric)

        for metric in metrics:
            # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly5_COOLING-linear\\"
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly5_COOLING-linear\\SA_"
            outfile_i = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "I" +"_" + metrics[metric] + "-1_INITIAL-poly5_COOLING-linear.graphml"
            outfile_f = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "F" +"_" + metrics[metric] + "-1_INITIAL-poly5_COOLING-linear.graphml"
            print(metric)
            experiment_poly5_linear(filename, outfile_i, outfile_f, stat_f, {metric: 1}, metric)

        for metric in metrics:
            # path = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly5_COOLING-quadratic\\"
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # outfile = "..\\..\\graph_drawings\\simulated_annealing\\" + metrics[metric] + "-1_INITIAL-poly5_COOLING-quadratic\\SA_"
            outfile_i = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "I" + "_" + metrics[metric] + "-1_INITIAL-poly5_COOLING-quadratic.graphml"
            outfile_f = "..\\..\\graph_drawings\\simulated_annealing\\experiment_1_drawings\\G" + str(i) + "F" + "_" + metrics[metric] + "-1_INITIAL-poly5_COOLING-quadratic.graphml"
            print(metric)
            experiment_poly5_quadratic(filename, outfile_i, outfile_f, stat_f, {metric: 1}, metric)



def evaluate():
    legend = "filename,EC,EO,AR,EL,GR,CA,Initial Config,Cooling Schedule,Initial Evaluation(Chosen Metrics),Final Evaluation(Chosen Metrics),Final Evaluation(All Metrics)"

    directory = os.fsencode(PATH)
    fnames = []
    for file in os.listdir(directory):
        fnames.append(os.fsdecode(file))

    #for graph_drawing in 



def main():
    stat_file = "..\\..\\data\\experiment_1.csv"
    with open(stat_file, "w") as stat_f:

        legend = "filename,EC,EO,AR,EL,GR,CA,Initial Config,Cooling Schedule,Initial Evaluation(Chosen Metrics),Initial Evaluation(All Metrics),Final Evaluation(Chosen Metrics),Final Evaluation(All Metrics),Num Crossings(Initial),Num Crossings(Final)\n"
        stat_f.write(legend)
    
    # Get file names
    directory = os.fsencode(PATH)
    fnames = []
    for file in os.listdir(directory):
        fnames.append(os.fsdecode(file))



    # Select random sample
    graphs = random.sample(fnames, 1)
    i = 1
    for graph in graphs:
        shutil.copyfile(PATH + graph, "..\\..\\graphs\\experiment_1\\G" + str(i) + "_" + graph)
        print(graph)
        experiment_metric_loop(graph, i)
        i += 1
        
    

    

if __name__ == "__main__":
    PATH = "..\\..\\graphs\\rome\\"
    all_metrics = {"edge_crossing": 1,
            "edge_orthogonality": 1,
            "angular_resolution": 1,
            "edge_length": 1,
            "gabriel_ratio": 1,
            "crossing_angle": 1,
    }
    main()