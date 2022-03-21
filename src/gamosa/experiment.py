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



def perform_sa(graph, metrics, initial, outfile_i, poly=0):

    ms = MetricsSuite(PATH + graph, metrics)
    sa = SimulatedAnnealing(ms, initial, "linear_a", 0, 100, 0.6, "random_bounded", 1, 1000, 60, 0, 0, n_polygon_sides=poly)
    ms.write_graph(outfile_i, sa.initial_config)
    G = sa.anneal()
    return {"i":sa.initial_config, "f":G}



def do_experiment(filename, outfile_i, outfile_f, stat_file, weights, initial_cfg, metrics):
    if initial_cfg[0:4] == "poly":
        GS = perform_sa(filename, weights, "polygon", outfile_i, poly=int(initial_cfg[-1]))
    else:
        GS = perform_sa(filename, weights, initial_cfg, outfile_i)

    ms_i = MetricsSuite(GS["i"], weights)
    ms_i2 = MetricsSuite(GS["i"], all_metrics)
    ms_G = MetricsSuite(GS["f"], weights)
    ms_G2 = MetricsSuite(GS["f"], all_metrics)
    ms_G.write_graph(outfile_f)
    
    line_list = [outfile_f[61]+outfile_f[62]+outfile_f[64:]]
    line_list.extend([str(m) for m in get_multi_metric_weights(metrics, weights)])
    line_list.append(initial_cfg)
    a = ms_i.combine_metrics()
    a2 = ms_i2.combine_metrics()
    b = ms_G.combine_metrics()
    c = ms_G2.combine_metrics()
    mi = [str(ms_i2.metrics[met]["value"]) if ms_i2.metrics[met]["value"] != None else "None"  for met in metrics]
    mf = [str(ms_G2.metrics[met]["value"]) if ms_G2.metrics[met]["value"] != None else "None"  for met in metrics]
    line_list.extend(mi)
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
    line_list.extend(mf)
    line_list.append(b)
    line_list.append(c)

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


def get_multi_metric_weights(metrics, weights):

    weight_list = []

    for metric_full in weights:
        weight_list.append(metrics[metric_full])

    metrics_short = ["EC", "EO", "AR", "EL", "GR"]

    number_list = []

    for short in metrics_short:
        if short in weight_list:
            number_list.append(1)
        else:
            number_list.append(0)

    return number_list


def get_multi_metric_fname(metrics, weights):
    weight_list = []

    for metric_full in weights:
        weight_list.append(metrics[metric_full])

    metrics_short = ["EC", "EO", "AR", "EL", "GR"]

    number_list = []

    for short in metrics_short:
        if short in weight_list:
            number_list.append(short + "-1")
        

    return "_".join(number_list)


def experiment_loop(filename, i):
    metrics = {"edge_crossing": "EC",
                "edge_orthogonality": "EO",
                "angular_resolution": "AR",
                "edge_length": "EL",
                "gabriel_ratio": "GR",
    }

    initials = ["random", "grid", "poly3", "poly5"]

    # experiments = {"1":{"edge_crossing":1, "angular_resolution":1},
    #             "2":{"edge_crossing":1, "edge_orthogonality":1},
    #             "3":{"edge_length":1, "edge_orthogonality":1},
    #             "4":{"edge_crossing":1, "gabriel_ratio":1},
    #             "5":{"angular_resolution":1, "gabriel_ratio":1},
    #             "6":{"edge_crossing":1, "angular_resolution":1, "gabriel_ratio":1},
    #             "7":{"edge_crossing":1, "edge_orthogonality":1, "edge_length":1},
    #             "8":{"edge_crossing":1, "angular_resolution":1, "edge_length":1, "gabriel_ratio":1}
    # }
    experiments = {"1":{"edge_crossing":1},
                "2":{"edge_orthogonality":1},
                "3":{"angular_resolution":1},
                "4":{"edge_length":1},
                "5":{"gabriel_ratio":1},
    }

    stat_file = "..\\..\\data\\experiment.csv"
    with open(stat_file, "a") as stat_f:

        # legend = "filename,EC,EO,AR,EL,GR,Initial Config,Initial Evaluation(Chosen Metrics),Initial Evaluation(All Metrics),Final Evaluation(Chosen Metrics),Final Evaluation(All Metrics)\n"
        # stat_f.write(legend)

        for exp in experiments:
            
            for cfg in initials:

                outfile_i = "..\\..\\graph_drawings\\simulated_annealing\\experiment_drawings\\G" + str(i) + "I" +"_" + get_multi_metric_fname(metrics, experiments[exp]) + "_INITIAL-" + cfg + ".graphml"
                outfile_f = "..\\..\\graph_drawings\\simulated_annealing\\experiment_drawings\\G" + str(i) + "F" +"_" + get_multi_metric_fname(metrics, experiments[exp]) + "_INITIAL-" + cfg + ".graphml"
                do_experiment(filename, outfile_i, outfile_f, stat_f, experiments[exp], cfg, metrics)
            



def main():
    stat_file = "..\\..\\data\\experiment.csv"
    with open(stat_file, "w") as stat_f:
        legend = "filename,EC,EO,AR,EL,GR,Initial Config,i_EC,i_EO,i_AR,i_EL,i_GR,Initial Evaluation(Chosen Metrics),Initial Evaluation(All Metrics),f_EC,f_EO,f_AR,f_EL,f_GR,Final Evaluation(Chosen Metrics),Final Evaluation(All Metrics)\n"
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
        shutil.copyfile(PATH + graph, "..\\..\\graphs\\experiment\\G" + str(i) + "_" + graph)
        print(graph)
        experiment_loop(graph, i)
        i += 1
        
    


if __name__ == "__main__":
    PATH = "..\\..\\graphs\\rome\\"
    all_metrics = {"edge_crossing": 1,
            "edge_orthogonality": 1,
            "angular_resolution": 1,
            "edge_length": 1,
            "gabriel_ratio": 1,
            #"crossing_angle": 1,
    }

    main()