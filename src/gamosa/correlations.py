from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import linregress


def get_correlations(filename, one_fig=True, hide_half=True, show_line=True, include_crossings=True):
    df = pd.read_csv(filename)
    df = df.drop(columns=['filename', 'SYM', 'time'])

    # Get rid of None valued entries (where there are > 250 crossings)
    df = df.loc[df['CA'] != "None"]
    df["CA"] = pd.to_numeric(df["CA"], downcast="float")

    df = df[:-2]

    
    metrics = ["EC", "EO", "NO", "AR", "NR", "EL", "GR", "CA"]
    if not include_crossings:
        df = df.drop(df[df['EC'] == 1].index)
        df = df.drop(columns=['NO'])
        metrics = ["EC", "EO", "AR", "NR", "EL", "GR", "CA"]

    axis_limit = len(metrics) - 1
    # Put every plot on one figure
    if one_fig:

        fig, axes = plt.subplots(axis_limit + 1, axis_limit + 1, sharey=True, sharex=True)
        
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
                
                # Only add axis labels to bottom and left plots
                if i == axis_limit:
                    axes[i,j].set(xlabel=metric1)

                if j == 0:
                    axes[i,j].set(ylabel=metric2)
            
                i += 1

            j += 1
            i = 0

        # Get rid of upper half of matrix of plots
        if hide_half:
            for i in range(axis_limit + 1):
                for j in range(axis_limit + 1):
                    if i <= j:
                        fig.delaxes(axes[i][j])
                        axes[i,j].set_axis_off()

        #plt.savefig("myImagePDF.pdf", format="pdf")
        plt.show()

    # Show each plot one after the other
    else:
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
    # Show correlation scatter plot for one pair of metrics
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


def correlation_matrix(filename, include_crossings=True):
    # Get the correlation matrix
    df = pd.read_csv(filename)
    df = df.drop(columns=['filename', 'SYM', 'time'])
    #df = df.drop(columns=['NO'])

    # Get rid of None valued entries
    df = df.loc[df['CA'] != "None"]
    df["CA"] = pd.to_numeric(df["CA"], downcast="float")

    df = df[:-2]

    labels = ["EC", "EO", "NO", "AR", "NR", "EL", "GR", "CA"]

    if not include_crossings:
        df = df.drop(df[df['EC'] == 1].index)
        df = df.drop(columns=['NO'])
        labels = ["EC", "EO", "AR", "NR", "EL", "GR", "CA"]


    corr_matrix = df.corr() #pearsons
    corr_matrix = round(corr_matrix, 3)

    fig, ax = plt.subplots()
    im = ax.imshow(corr_matrix)
    im.set_clim(-1, 1)
    ax.grid(False)
    ax.xaxis.set(ticks=range(len(labels)), ticklabels=labels)
    ax.yaxis.set(ticks=range(len(labels)), ticklabels=labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(corr_matrix.iloc[i,j]), ha='center', va='center', color='red')
            pass
    cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
    #plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def main():
    filename = "..\\..\\data\\nathan_distributions_all.csv"
    correlation_matrix(filename, include_crossings=False)
    #get_correlations(filename, True, include_crossings=False)
    #get_correlation(filename, "AR", "NR")
    #get_distributions(filename)
    

if __name__ == "__main__":
    main()