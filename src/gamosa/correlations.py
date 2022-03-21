from unicodedata import name
from metrics_suite import MetricsSuite
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import pandas as pd
import random

def get_correlations(filename, one_fig=True):
    df = pd.read_csv(filename)
    df = df.drop(columns=['filename', 'SYM', 'time'])

    # Get rid of None valued entries
    df = df.loc[df['CA'] != "None"]
    df["CA"] = pd.to_numeric(df["CA"], downcast="float")

    df = df[:-2]

    if one_fig:
        # fig, axes = plt.subplots(8, 8, sharey=True)
        # metrics = ["EC", "EO", "NO", "AR", "NR", "EL", "GR", "CA"]
        # covered = []
        # i = 0
        # j = 0
        # for metric1 in metrics:
        #     for metric2 in metrics:
        #         if (metric2, metric1) in covered:
        #             continue

        #         covered.append((metric1, metric2))

        #         axes[j,i].scatter(df[metric1], df[metric2])
        #         axes[j,i].set(xlim=(0,1), ylim=(0,1))

        #         if i == 7 - j:
        #             axes[j,i].set(xlabel=metric1)
        #         else:
        #             axes[7-i,7-j].set_axis_off()

        #         if j == 0:
        #             axes[i,j].set(ylabel=metric2)
            
        #         i += 1

        #     j += 1
        #     i = 0

        fig, axes = plt.subplots(8, 8, sharey=True, sharex=True)
        metrics = ["EC", "EO", "NO", "AR", "NR", "EL", "GR", "CA"]
        i = 0
        j = 0
        for metric1 in metrics:
            for metric2 in metrics:

                axes[i,j].scatter(df[metric1], df[metric2])
                axes[i,j].set(xlim=(0,1), ylim=(0,1))

                if i == 7:
                    axes[i,j].set(xlabel=metric1)

                if j == 0:
                    axes[i,j].set(ylabel=metric2)
            
                i += 1

            j += 1
            i = 0

        for i in range(8):
            for j in range(8):
                if i <= j:
                    fig.delaxes(axes[i][j])
                    axes[i,j].set_axis_off()

        plt.show()

    else:
        metrics = ["EC", "EO", "NO", "AR", "NR", "EL", "GR", "CA"]
        covered = []
        for metric1 in metrics:
            for metric2 in metrics:
                if metric1 == metric2 or (metric2, metric1) in covered:
                    continue

                covered.append((metric1, metric2))

                plt.scatter(df[metric1], df[metric2])
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.xlabel(metric1)
                plt.ylabel(metric2)
                
                plt.show()


def get_correlation(filename, metric1, metric2):
    df = pd.read_csv(filename)
    df = df.drop(columns=['filename', 'SYM', 'time'])

    # Get rid of None valued entries
    df = df.loc[df['CA'] != "None"]
    df["CA"] = pd.to_numeric(df["CA"], downcast="float")

    df = df[:-2]

    plt.scatter(df[metric1], df[metric2])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(metric1)
    plt.ylabel(metric2)
    
    plt.show()


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


def main():
    filename = "..\\..\\data\\nathan_distributions_all.csv"
    #get_correlation(filename, "AR", "NR")
    get_correlations(filename, True)
    
    #get_distributions(filename)
    

if __name__ == "__main__":
    main()