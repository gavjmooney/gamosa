from metrics_suite import MetricsSuite
from matplotlib import pyplot as plt
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

    for col in df:
        
        df.hist(column=col, bins=40, figsize=(10,8))
        plt.ylim(0,32180)
        
        plt.show()
        
        # df.hist(column=col, bins=40)
    

    fig, axs = plt.subplots(ncols=2, nrows=4)
    #hist = df.hist(bins=20)
    #hist1 = df.hist(bins=20, sharex=True)
    hist2 = df.hist(bins=40, sharex=True, sharey=True, ax=axs)
    

    plt.subplots_adjust(left=0.3,right=0.7,bottom=0.03,top=0.97,wspace=0.25,hspace=0.25)

    plt.show()

def get_sym_distributions(filename):
    df = pd.read_csv(filename)
    #df = df.drop(columns=['filename', 'SYM', 'CA', 'time'])
    df = df.drop(columns=['filename', 'time'])

    # Get rid of None valued entries
    # df = df.loc[df['CA'] != "None"]
    # df["CA"] = pd.to_numeric(df["CA"], downcast="float")

    for col in df:
        
        df.hist(column=col, bins=40, figsize=(10,8), range=[0,1])
        #plt.ylim(0,32180)
        #plt.xticks([x/10 for x in range(11)])
        if col == "SYM":
            plt.ylabel("# of Graph Drawings")
            plt.xlabel("Metric Value")
            plt.title("Distribution of SYM Values")
            plt.savefig("sym.pdf", format="pdf", bbox_inches="tight")
        
            plt.show()
        
        # df.hist(column=col, bins=40)
    

    fig, axs = plt.subplots()
    #hist = df.hist(bins=20)
    #hist1 = df.hist(bins=20, sharex=True)
    hist2 = df.hist(bins=40, sharex=True, sharey=False, ax=axs)
    

    plt.subplots_adjust(left=0.3,right=0.7,bottom=0.03,top=0.97,wspace=0.25,hspace=0.25)

    plt.show()


def get_distributions_excluding_no_crossings(filename):
    df = pd.read_csv(filename)
    #df = df.drop(columns=['filename', 'SYM', 'CA', 'time'])
    df = df.drop(columns=['filename', 'SYM', 'time'])

    # Get rid of None valued entries
    df = df.loc[df['CA'] != "None"]
    df["CA"] = pd.to_numeric(df["CA"], downcast="float")

    df = df.drop(df[df['EC'] == 1].index)
    # df = df.drop(columns=['NO'])

    # print(len(df))
    # for col in df:
        
    #     df.hist(column=col, bins=40, figsize=(10,8))
    #     plt.ylim(0,32180)
        
    #     plt.show()
        
        # df.hist(column=col, bins=40)
    

    fig, axs = plt.subplots(ncols=2, nrows=4)
    #hist = df.hist(bins=20)
    #hist1 = df.hist(bins=20, sharex=True)
    hist2 = df.hist(bins=40, sharex=True, sharey=True, ax=axs)
    

    plt.subplots_adjust(left=0.3,right=0.7,bottom=0.03,top=0.97,wspace=0.25,hspace=0.25)

    plt.show()


def count_ca(filename):
    # Number of actual graphs for which crossing angle was calcuated
    df = pd.read_csv(filename)
    print(len(df[(df['CA'] != "None")]))


def combine_ca(filename1, filename2, outfile):
    # Old function which combined two CSV files. The first had data for drawings with les than 100 crossings
    # and the second had those for crossings between 100-250
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
    # Old function which combined two CSV files. The first had data for drawings with les than 100 crossings
    # and the second had those for crossings between 100-250
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
    """Function for getting the metric values for a directory(in_dir) of graph drawings."""
    # Change this to decide which metrics to calculate
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
    # This is identical to get_metric_vals, just adapted so it can run repeatedly on the slower symmetry calculations.
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

    with open(out_file, "a") as out_f:
        # header = "filename,EC,EO,NO,AR,SYM,NR,EL,GR,CA,crossings,time\n"
        # out_f.write(header)
    
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

        for file in graphs:
            if i == limit:
                break
            i += 1
            time_a = time.time()

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
                        ms.metrics["crossing_angle"]["value"],
                        ms.metrics["edge_crossing"]["num_crossings"]
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
    with open(stat_file, "a") as out_f:
        out_f.write("FIXED\n\n")
        out_f.write(summary)

def combine_sym():
    filename1 = "..\\..\\data\\nathan_sym_distributions.csv"
    filename2 = "..\\..\\data\\nathan_distributions.csv"

    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)

    print(df1)
    df1 = df1.drop(df1[df1['num_crossings'] >= 50].index)
    df1 = df1.drop_duplicates()

    df1 = df1[:-2]

    df3 = pd.DataFrame(columns=['filename', 'EC', 'EO', 'NO', 'AR', 'SYM', 'NR', 'EL', 'GR', 'CA','time'])
    print(df1)
    new_col = []

    for index, row in df2.iterrows():
        #assert row['filename'] == df1.iloc[index]['filename']
        for index2, row2 in df1.iterrows():
            if row['filename'] == row2['filename']:
                row_copy = row.copy()
                row_copy['SYM'] = row2['SYM']
                df3 = df3.append(row_copy)
                
    print(df3)
    print(df1['time'].mean())
    print(df1['time'].max())
    df3.to_csv("..\\..\\data\\nathan_sym_distributions_all.csv")

def main():
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

    # in_dir = "..\\..\\graph_drawings\\nathan\\"
    # out_file = "..\\..\\data\\nathan_sym_distributions.csv"
    # stat_file = "..\\..\\data\\nathan_sym_stats.txt"

    # get_sym(in_dir, out_file, stat_file, file_type="GraphML")

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

    #filename = "..\\..\\data\\nathan_distributions_all_copy.csv"
    #get_distributions_excluding_no_crossings(filename)

    #combine_sym()
    filename = "..\\..\\data\\nathan_sym_distributions_all.csv"
    get_sym_distributions(filename)


if __name__ == "__main__":

    main()
    