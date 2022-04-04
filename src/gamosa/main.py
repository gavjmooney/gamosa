from metrics_suite import MetricsSuite
from simulated_annealing import SimulatedAnnealing
import os


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


def simulated_annealing_example(filename, metric_weights):
    ms = MetricsSuite(filename, metric_weights)

    sa = SimulatedAnnealing(ms, initial_config="polygon", cooling_schedule="quadratic_a", n_polygon_sides=6)
    
    initial_graph_drawing = sa.initial_config
    ms_initial = MetricsSuite(initial_graph_drawing, metric_weights)
    
    annealed_graph_drawing = sa.anneal()
    ms_annealed = MetricsSuite(annealed_graph_drawing, metric_weights)

    ms_initial.draw_graph()
    ms_initial.pretty_print_metrics()
    
    ms_annealed.draw_graph()
    ms_annealed.pretty_print_metrics()


def main():
    # Change Me
    dir = "..\\..\\graphs\\north\\"
    output_file = "..\\..\\data\\test.csv"
    metric_weights = {"edge_crossing": 1,
                "edge_orthogonality": 0,
                "node_orthogonality": 0,
                "angular_resolution": 0,
                "symmetry": 0,
                "node_resolution": 0,
                "edge_length": 0,
                "gabriel_ratio": 1,
                "crossing_angle": 0,
    }

    get_distributions(dir, output_file, metric_weights)
    simulated_annealing_example("..\\..\\graphs\\rome\\grafo147.29.graphml", metric_weights)


if __name__ == "__main__":
    main()    