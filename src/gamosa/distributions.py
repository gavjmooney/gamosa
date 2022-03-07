from metrics_suite import MetricsSuite
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import pandas as pd
import random

def get_distributions(filename):
    df = pd.read_csv(filename)
    #df = df.drop(columns=['filename', 'SYM', 'CA', 'time'])
    df = df.drop(columns=['filename', 'SYM', 'time'])

    # Get rid of None valued entries
    df = df.loc[df['CA'] != "None"]
    df["CA"] = pd.to_numeric(df["CA"], downcast="float")


    # for col in df:
    #     print(col)
    #     df.hist(column=col)

    #hist = df.hist(bins=20)
    #hist1 = df.hist(bins=20, sharex=True)
    hist2 = df.hist(bins=20, sharex=True, sharey=True)
    
    # hist4 = df.hist(bins=20, column="CA")
    
    plt.show()


def count_ca(filename):
    df = pd.read_csv(filename)

    print(len(df[(df['CA'] != "None")]))



def combine_ca(filename1, filename2, outfile):
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)

    new_col = []

    for index, row in df1.iterrows():
        assert row['EC'] == df2.iloc[index]['EC']
        if df2.iloc[index]['CA'] != "None":
            new_col.append(df2.iloc[index]['CA'])

        elif row['CA'] != "None":
            new_col.append(row['CA'])
        else:
            new_col.append("None")
            

    df3 = df1.copy()

    df3.drop(columns=['CA'])
    df3['CA'] = new_col

    df3.to_csv(outfile)

def combine_all():
    filename1 = "..\\..\\data\\nathan_distributions.csv"
    filename2 = "..\\..\\data\\nathan_crossing_distributions_less250.csv"
    filename3 = "..\\..\\data\\nathan_distributions_all.csv"

    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)

    df3 = df1.copy()
    df3.drop(columns=['CA'])
    df3['CA'] = df2['CA']

    df3.to_csv(filename3)



def get_metric_vals(in_dir, out_file, stat_file, file_type="GraphML"):
    weights = {"edge_crossing": 1,
                "edge_orthogonality": 0,
                "node_orthogonality": 0,
                "angular_resolution": 0,
                "symmetry": 0,
                "node_resolution": 0,
                "edge_length": 0,
                "gabriel_ratio": 0,
                "crossing_angle": 1,
    } 

    with open(out_file, "w") as out_f:
        header = "filename,EC,EO,NO,AR,SYM,NR,EL,GR,CA,time\n"
        out_f.write(header)
    
    
        directory = os.fsencode(in_dir)
        i = 0

        total_nodes = 0
        total_edges = 0
        total_time = 0

        for file in os.listdir(directory):
            i += 1
            time_a = time.time()
            filename = os.fsdecode(file)
            print(f"{i}:\t{filename}\t{i/32190*100:.2f}%")


            ms = MetricsSuite(in_dir + filename, weights, file_type=file_type)
            ms.calculate_metrics()

            total_nodes += ms.graph.number_of_nodes()
            total_edges += ms.graph.number_of_edges()

            metrics = [ms.metrics["edge_crossing"]["value"],
                        ms.metrics["edge_orthogonality"]["value"],
                        ms.metrics["node_orthogonality"]["value"],
                        ms.metrics["angular_resolution"]["value"],
                        ms.metrics["symmetry"]["value"],
                        ms.metrics["node_resolution"]["value"],
                        ms.metrics["edge_length"]["value"],
                        ms.metrics["gabriel_ratio"]["value"],
                        ms.metrics["crossing_angle"]["value"]
            ]
            
            time_b = time.time()
            time_taken = time_b - time_a
            total_time += time_taken
            print(f"{i}: {time_taken}")
            # if v > 0.001 else "<0.001"
            line = filename + "," + ",".join(str(round(v, 3)) if v != None else "None" for v in metrics) + f",{round(time_taken,3)}\n"
            out_f.write(line)


    summary = f"Average nodes: {total_nodes/i}\nAverage edges: {total_edges/i}\nAverage time: {total_time/i}"
    print(summary)
    with open(stat_file, "w") as out_f:
        out_f.write(summary)


def get_sym(in_dir, out_file, stat_file, file_type="GraphML"):
    weights = {"edge_crossing": 1,
                "edge_orthogonality": 0,
                "node_orthogonality": 0,
                "angular_resolution": 0,
                "symmetry": 1,
                "node_resolution": 0,
                "edge_length": 0,
                "gabriel_ratio": 0,
                "crossing_angle": 0,
    } 

    with open(out_file, "w") as out_f:
        header = "filename,EC,EO,NO,AR,SYM,NR,EL,GR,CA,time\n"
        out_f.write(header)
    
    
        i = 0

        total_nodes = 0
        total_edges = 0
        total_time = 0

        directory = os.fsencode(in_dir)
        fnames = []
        for file in os.listdir(directory):
            fnames.append(os.fsdecode(file))


        limit = 10

        # Select random sample
        graphs = random.sample(fnames, limit)

        #for file in os.listdir(directory):
        for file in graphs:
            if i == limit:
                break
            i += 1
            time_a = time.time()
            #filename = os.fsdecode(file)
            filename = file
            print(f"{i}:\t{filename}\t{i/limit*100:.2f}%")


            ms = MetricsSuite(in_dir + filename, weights, file_type=file_type)
            ms.calculate_metrics()

            total_nodes += ms.graph.number_of_nodes()
            total_edges += ms.graph.number_of_edges()

            metrics = [ms.metrics["edge_crossing"]["value"],
                        ms.metrics["edge_orthogonality"]["value"],
                        ms.metrics["node_orthogonality"]["value"],
                        ms.metrics["angular_resolution"]["value"],
                        ms.metrics["symmetry"]["value"],
                        ms.metrics["node_resolution"]["value"],
                        ms.metrics["edge_length"]["value"],
                        ms.metrics["gabriel_ratio"]["value"],
                        ms.metrics["crossing_angle"]["value"]
            ]
            
            time_b = time.time()
            time_taken = time_b - time_a
            total_time += time_taken
            print(f"{i}: {time_taken}")
            # if v > 0.001 else "<0.001"
            line = filename + "," + ",".join(str(round(v, 3)) if v != None else "None" for v in metrics) + f",{round(time_taken,3)}\n"
            out_f.write(line)


    summary = f"Average nodes: {total_nodes/i}\nAverage edges: {total_edges/i}\nAverage time: {total_time/i}"
    print(summary)
    with open(stat_file, "w") as out_f:
        out_f.write(summary)

def main():
    # in_dir = "..\\..\\graph_drawings\\nathan\\FR_1\\"
    # out_file = "..\\..\\data\\nathan_fr_1_distributions.csv"
    # stat_file = "..\\..\\data\\nathan_fr_1_stats.txt"

    # in_dir = "..\\..\\graph_drawings\\nathan\\KK_1\\"
    # out_file = "..\\..\\data\\nathan_kk_1_distributions.csv"
    # stat_file = "..\\..\\data\\nathan_kk_1_stats.txt"

    # in_dir = "..\\..\\graph_drawings\\nathan\\HOLA_1\\"
    # out_file = "..\\..\\data\\nathan_hola_1_distributions.csv"
    # stat_file = "..\\..\\data\\nathan_hola_1_stats.txt"

    # in_dir = "..\\..\\graph_drawings\\nathan\\SUGI_1\\"
    # out_file = "..\\..\\data\\nathan_sugi_1_distributions.csv"
    # stat_file = "..\\..\\data\\nathan_sugi_1_stats.txt"

    # in_dir = "..\\..\\graph_drawings\\nathan\\HOLA_0\\"
    # out_file = "..\\..\\data\\nathan_hola_0_distributions.csv"
    # stat_file = "..\\..\\data\\nathan_hola_0_stats.txt"

    # in_dir = "..\\..\\graph_drawings\\nathan\\FR_0\\"
    # out_file = "..\\..\\data\\nathan_fr_0_distributions.csv"
    # stat_file = "..\\..\\data\\nathan_fr_0_stats.txt"

    # in_dir = "..\\..\\graph_drawings\\nathan\\"
    # out_file = "..\\..\\data\\nathan_distributions.csv"
    # stat_file = "..\\..\\data\\nathan_stats.txt"

    # in_dir = "..\\..\\graph_drawings\\asonam\\"
    # out_file = "..\\..\\data\\asonam_b-f_distributions.csv"
    # stat_file = "..\\..\\data\\asonam_b-f_stats.txt"

    #in_dir = "..\\..\\graph_drawings\\nathan\\"
    #out_file = "..\\..\\data\\nathan_crossing_distributions_100-250.csv"
    #stat_file = "..\\..\\data\\nathan_crossing_stats_100-250.txt"

    #get_metric_vals(in_dir, out_file, stat_file, file_type="GraphML")

    in_dir = "..\\..\\graph_drawings\\nathan\\"
    out_file = "..\\..\\data\\nathan_sym_distributions.csv"
    stat_file = "..\\..\\data\\nathan_sym_stats.txt"

    get_sym(in_dir, out_file, stat_file, file_type="GraphML")

    #filename = "..\\..\\data\\nathan_distributions.csv"
    #filename = "..\\..\\data\\asonam_b-f_distributions.csv"

    #filename = "..\\..\\data\\nathan_crossing_distributions.csv"
    #get_distributions(filename)

    
    # filename1 = "..\\..\\data\\nathan_crossing_distributions_less100.csv"
    # filename2 = "..\\..\\data\\nathan_crossing_distributions_100-250.csv"
    # filename3 = "..\\..\\data\\nathan_crossing_distributions_less250.csv"

    #combine_ca(filename1, filename2, filename3)
    # count_ca(filename1)
    # count_ca(filename2)
    # count_ca(filename3)

    #29261 total with <250

    # combine_all()

    # filename = "..\\..\\data\\nathan_distributions_all.csv"
    # get_distributions(filename)
    


if __name__ == "__main__":

    main()
    