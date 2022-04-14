from metrics_suite import MetricsSuite
from simulated_annealing import SimulatedAnnealing
import os
import networkx as nx
import matplotlib.pyplot as plt


def get_distributions(input_directory, output_file, metric_weights):

    with open(output_file, "w") as out_f:
        
        legend = "filename,EC,EO,NO,AR,SYM,NR,EL,GR,CA,num_crossings\n"
        out_f.write(legend)

        i = 0
        directory = os.fsencode(input_directory)
        for file in os.listdir(directory):
            percent_done = i/len(os.listdir(directory))*100
            if i % 10 == 0:
                print(f"{percent_done:.2f}%")

            filename = os.fsdecode(file)
            
            
            ms = MetricsSuite(input_directory + filename, metric_weights)
            ms.calculate_metrics()
            #ms.pretty_print_metrics()
            
            values = [ms.metrics["edge_crossing"]["value"],
                        ms.metrics["edge_orthogonality"]["value"],
                        ms.metrics["node_orthogonality"]["value"],
                        ms.metrics["angular_resolution"]["value"],
                        ms.metrics["symmetry"]["value"],
                        ms.metrics["node_resolution"]["value"],
                        ms.metrics["edge_length"]["value"],
                        ms.metrics["gabriel_ratio"]["value"],
                        ms.metrics["crossing_angle"]["value"],
                        ms.metrics["edge_crossing"]["num_crossings"],
            ]

            line = filename + "," + ",".join(str(round(v, 3)) if v != None else "None" for v in values) + "\n"
            out_f.write(line)

            i += 1


def simulated_annealing_example(filename, metric_weights, output_directory):
    ms = MetricsSuite(filename, metric_weights)

    sa = SimulatedAnnealing(ms, initial_config="polygon", cooling_schedule="quadratic_a", n_polygon_sides=6)
    
    initial_graph_drawing = sa.initial_config
    ms_initial = MetricsSuite(initial_graph_drawing, metric_weights)
    
    annealed_graph_drawing = sa.anneal()
    ms_annealed = MetricsSuite(annealed_graph_drawing, metric_weights)

    #ms_initial.draw_graph()
    ms_initial.write_graph(output_directory + "SA_initial_" + filename.split("\\")[-1])
    ms_initial.pretty_print_metrics()
    
    #ms_annealed.draw_graph()
    ms_annealed.write_graph(output_directory + "SA_final_" + filename.split("\\")[-1])
    ms_annealed.pretty_print_metrics()


def report_graphs():
    PATH = "..\\..\\graph_drawings\\report\\"

    gds = ["random", "fr", "initial", "GR", "EC-GR", "EL-EO", "EC-AR-GR", "EC-AR-EL-GR"]

    graphs = [nx.complete_graph(20), nx.complete_bipartite_graph(5,5), nx.cycle_graph(10), nx.circular_ladder_graph(10),
     nx.grid_2d_graph(5,5), nx.cubical_graph(), nx.dodecahedral_graph(), nx.balanced_tree(2,6), nx.balanced_tree(2,3)]

    g_names = ["complete20", "complete5,5", "cycle10", "circ_ladder10", "grid5,5", "cube", "dodecahedron", "tree2,6", "tree2,3"]

    for g, name in zip(graphs, g_names):
        G = g.copy()
        # nx.draw(g)
        # print(f"n={g.number_of_nodes()}, m={g.number_of_edges()}")
        # plt.show()

        pos = nx.random_layout(G)
        for k,v in pos.items():
            pos[k] = {"x":v[0]*G.number_of_nodes()*20, "y":v[1]*G.number_of_nodes()*20}

        nx.set_node_attributes(G, pos)

        ms = MetricsSuite(G)
        #ms.write_graph(PATH + "random_" + name + ".graphml")

        ####################

        pos = nx.fruchterman_reingold_layout(G)
        for k,v in pos.items():
            pos[k] = {"x":v[0]*G.number_of_nodes()*20, "y":v[1]*G.number_of_nodes()*20}

        nx.set_node_attributes(G, pos)

        ms = MetricsSuite(G)
        #ms.write_graph(PATH + "fr_" + name + ".graphml")

        ####################

        #ms = MetricsSuite(G, {"gabriel_ratio":1})
        #ms = MetricsSuite(G, {"edge_crossing":1, "gabriel_ratio":1})
        #ms = MetricsSuite(G, {"edge_length":1, "edge_orthogonality":1})
        #ms = MetricsSuite(G, {"edge_crossing":1, "angular_resolution":1, "gabriel_ratio":1})
        ms = MetricsSuite(G, {"edge_crossing":1, "angular_resolution":1, "edge_length":1, "gabriel_ratio":1})
        
        sa = SimulatedAnnealing(ms, "polygon", n_polygon_sides=5)

        #ms.write_graph(PATH + "initial_" + name + ".graphml", sa.initial_config)

        #ms.write_graph(PATH + "GR_" + name + ".graphml", sa.anneal())
        #ms.write_graph(PATH + "EC-GR_" + name + ".graphml", sa.anneal(True))
        #ms.write_graph(PATH + "EL-EO_" + name + ".graphml", sa.anneal())
        #ms.write_graph(PATH + "EC-AR-GR_" + name + ".graphml", sa.anneal(True))
        ms.write_graph(PATH + "EC-AR-EL-GR_" + name + ".graphml", sa.anneal(True))
        

def main():
    # Change Me
    dir = "..\\..\\graphs\\north\\"
    output_file = "..\\..\\data\\test.csv"
    metric_weights = {"edge_crossing": 1,
                "edge_orthogonality": 0,
                "node_orthogonality": 0,
                "angular_resolution": 0,
                "symmetry": 0,
                "node_resolution": 1,
                "edge_length": 0,
                "gabriel_ratio": 1,
                "crossing_angle": 0,
    }

    sa_output_directory = "..\\..\\graph_drawings\\"
    #get_distributions(dir, output_file, metric_weights)
    simulated_annealing_example("..\\..\\graphs\\rome\\grafo147.29.graphml", metric_weights, sa_output_directory)
    #report_graphs()


if __name__ == "__main__":
    main()    