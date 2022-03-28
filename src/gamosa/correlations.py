from turtle import color
from unicodedata import name
from metrics_suite import MetricsSuite
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import pandas as pd
import random
from scipy.stats import pearsonr, spearmanr, linregress

def get_correlations(filename, one_fig=True, hide_half=True, show_line=True):
    df = pd.read_csv(filename)
    df = df.drop(columns=['filename', 'SYM', 'time'])

    # Get rid of None valued entries
    df = df.loc[df['CA'] != "None"]
    df["CA"] = pd.to_numeric(df["CA"], downcast="float")

    df = df[:-2]

    if one_fig:

        fig, axes = plt.subplots(8, 8, sharey=True, sharex=True)
        metrics = ["EC", "EO", "NO", "AR", "NR", "EL", "GR", "CA"]
        i = 0
        j = 0
        for metric1 in metrics:
            for metric2 in metrics:

                axes[i,j].scatter(df[metric1], df[metric2], s=5)
                #axes[i,j].plot(df[metric1], df[metric2], linewidth=0, marker='o',label='data', s=5)
                
                axes[i,j].set(xlim=(0,1), ylim=(0,1))
                
                if show_line:
                    slope, intercept, r, p, stderr = linregress(df[metric1], df[metric2])
                    axes[i,j].plot(df[metric1], intercept + slope * df[metric1], linewidth=1, color='red')

                if i == 7:
                    axes[i,j].set(xlabel=metric1)

                if j == 0:
                    axes[i,j].set(ylabel=metric2)
            
                i += 1

            j += 1
            i = 0

        if hide_half:
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

                plt.scatter(df[metric1], df[metric2], s=0.1)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                if show_line:
                    slope, intercept, r, p, stderr = linregress(df[metric1], df[metric2])
                    plt.plot(df[metric1], intercept + slope * df[metric1], linewidth=1, color='red')
                plt.xlabel(metric1)
                plt.ylabel(metric2)
                
                plt.show()


def get_correlation(filename, metric1, metric2, show_line=True):
    df = pd.read_csv(filename)
    df = df.drop(columns=['filename', 'SYM', 'time'])

    # Get rid of None valued entries
    df = df.loc[df['CA'] != "None"]
    df["CA"] = pd.to_numeric(df["CA"], downcast="float")

    df = df[:-2]

    plt.scatter(df[metric1], df[metric2])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if show_line:
        slope, intercept, r, p, stderr = linregress(df[metric1], df[metric2])
        plt.plot(df[metric1], intercept + slope * df[metric1], linewidth=1, color='red')
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



def correlation_matrix(filename):
    df = pd.read_csv(filename)
    df = df.drop(columns=['filename', 'SYM', 'time'])

    # Get rid of None valued entries
    df = df.loc[df['CA'] != "None"]
    df["CA"] = pd.to_numeric(df["CA"], downcast="float")

    df = df[:-2]

    #print(np.cov(df['EC'], df['EO']))
    # corr, _ = pearsonr(df['EC'], df['EO'])
    # print('Pearsons correlation coeeficient: %.3f' % corr) #scipy
    # # print(df["EC"].corr(df["EO"])) #pearsons with pandas

    # corr, _ = spearmanr(df['EC'], df['EO'])
    # print('Spearmans correlation coeeficient: %.3f' % corr)

    #print(np.corrcoef(df['EC'], df['EO'], df['NO'], df['AR'], df['NR'], df['EL'], df['GR'], df['CA']))
    # result = linregress(df['EC'], df['EO'])
    # print(result.slope)
    # print(result.intercept)
    # # print(result.rvalue)
    # # print(result.pvalue)
    # # print(result.stderr)
    # print()
    
    labels = ["EC", "EO", "NO", "AR", "NR", "EL", "GR", "CA"]

    corr_matrix = df.corr() #pearsons
    corr_matrix = round(corr_matrix, 3)

    fig, ax = plt.subplots()
    im = ax.imshow(corr_matrix)
    im.set_clim(-1, 1)
    ax.grid(False)
    ax.xaxis.set(ticks=range(len(labels)), ticklabels=labels)
    ax.yaxis.set(ticks=range(len(labels)), ticklabels=labels)
    #ax.set_ylim(2.5, -0.5)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, str(corr_matrix.iloc[i,j]), ha='center', va='center', color='red')
            pass
    cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
    plt.show()



def main():
    filename = "..\\..\\data\\nathan_distributions_all.csv"
    correlation_matrix(filename)
    get_correlations(filename, True)
    #get_correlation(filename, "AR", "NR")

    #get_distributions(filename)
    

if __name__ == "__main__":
    main()