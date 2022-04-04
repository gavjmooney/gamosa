from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def count_graphs():
    file = "..\\..\\data\\nathan_distributions_all_copy.csv"
    df = pd.read_csv(file)

    print(f"Total graphs: {len(df)}")
    print(f"Total graphs with less than 250 crossings: {len(df[df['CA'] != 'None'])}")

    file = "..\\..\\data\\nathan_crossing_distributions_less100.csv"
    df = pd.read_csv(file)

    print(f"Total graphs with less than 100 crossings: {len(df[df['CA'] != 'None'])}")

    file = "..\\..\\data\\nathan_crossing_distributions_100-250.csv"
    df = pd.read_csv(file)

    print(f"Total graphs with 100-250 crossings: {len(df[df['CA'] != 'None'])}")

    file = "..\\..\\data\\nathan_crossing_distributions_less250.csv"
    df = pd.read_csv(file)

    print(f"Total graphs with less than 250 crossings: {len(df[df['CA'] != 'None'])}")

    print(23661+5600-29180)

def load_file(filename):

    df = pd.read_csv(filename)

    # Convert to float
    df['Final Evaluation(Chosen Metrics)'] = pd.to_numeric(df['Final Evaluation(Chosen Metrics)'], downcast='float')
    df['Initial Evaluation(Chosen Metrics)'] = pd.to_numeric(df['Initial Evaluation(Chosen Metrics)'], downcast='float')
    df['Final Evaluation(All Metrics)'] = pd.to_numeric(df['Final Evaluation(All Metrics)'], downcast='float')
    df['Initial Evaluation(All Metrics)'] = pd.to_numeric(df['Initial Evaluation(All Metrics)'], downcast='float')

    # Calculate improvments in metrics
    df['all_diff'] = df['Final Evaluation(All Metrics)'] - df['Initial Evaluation(All Metrics)']
    df['chosen_diff'] = df['Final Evaluation(Chosen Metrics)'] - df['Initial Evaluation(Chosen Metrics)']

    df['EC_diff'] = df['f_EC'] - df['i_EC']
    df['EO_diff'] = df['f_EO'] - df['i_EO']
    df['AR_diff'] = df['f_AR'] - df['i_AR']
    df['EL_diff'] = df['f_EL'] - df['i_EL']
    df['GR_diff'] = df['f_GR'] - df['i_GR']

    return df    


def compare_before_after(df):

    EC_df = df[df['EC'] == 1]
    EO_df = df[df['EO'] == 1]
    AR_df = df[df['AR'] == 1]
    EL_df = df[df['EL'] == 1]
    GR_df = df[df['GR'] == 1]

    averages_before = [round(x,3) for x in [df["i_EC"].mean(), df["i_EO"].mean(), df["i_AR"].mean(), df["i_EL"].mean(), df["i_GR"].mean()]] #bar
    averages_after = [round(x,3) for x in [df["f_EC"].mean(), df["f_EO"].mean(), df["f_AR"].mean(), df["f_EL"].mean(), df["f_GR"].mean()]]#bar

    both = [averages_before, averages_after]

    chosen_before = round(df['Initial Evaluation(Chosen Metrics)'].mean(),3) #line
    chosen_after = round(df['Final Evaluation(Chosen Metrics)'].mean(),3) #line

    all_before = round(df['Initial Evaluation(All Metrics)'].mean(),3) #dotted line
    all_after = round(df['Final Evaluation(All Metrics)'].mean(),3) #dotted line

    fig, ax = plt.subplots()

    labels = ["EC", "EO", "AR", "EL", "GR"]
    y_size = [x / 10 for x in range(11)]

    x = np.arange(len(labels))
    width = 0.35

    before = ax.bar(x - width/2, averages_before, width, label="Before SA")
    after = ax.bar(x + width/2, averages_after, width, label="After SA")

    line_chosen_before = ax.axhline(y=chosen_before, color="blue", linestyle='solid', label="Chosen Metrics before SA")
    line_chosen_after = ax.axhline(y=chosen_after, color="orange", linestyle='solid', label="Chosen Metrics after SA")

    line_all_before = ax.axhline(y=all_before, color="blue", linestyle='dashed', label="All Metrics before SA")
    line_all_after = ax.axhline(y=all_after, color="orange", linestyle='dashed', label="All Metrics after SA")

    ax.set_xticks(x, labels)
    ax.legend()

    ax.set_yticks(y_size)
    ax.set_ylabel("Metric Value")
    ax.set_xlabel("Metrics")

    # ax.bar_label(before, padding=3)
    # ax.bar_label(after, padding=3)
    
    ax.set_title("Metric Improvements")
    plt.show()

def compare_before_after2(df):

    EC_df = df[df['EC'] == 1]
    EO_df = df[df['EO'] == 1]
    AR_df = df[df['AR'] == 1]
    EL_df = df[df['EL'] == 1]
    GR_df = df[df['GR'] == 1]

    dfs = [EC_df, EO_df, AR_df, EL_df, GR_df]


    i = 0
    skip = False
    fig, axs = plt.subplots(2,3)
    for ax in axs.flat:
        if i == 2:
            i += 1

            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            handles, labels = plt.gca().get_legend_handles_labels()

            line1 = Line2D([0], [0], color="blue",linewidth=1, linestyle="solid", label="Chosen Metrics before SA")
            line2 = Line2D([0], [0], color="orange",linewidth=1, linestyle="solid", label="Chosen Metrics after SA")
            line3 = Line2D([0], [0], color="blue",linewidth=1, linestyle="dashed", label="All Metrics before SA")
            line4 = Line2D([0], [0], color="orange",linewidth=1, linestyle="dashed", label="All Metrics after SA")

            patch1 = mpatches.Patch(color="tab:blue", label="Before SA")
            patch2 = mpatches.Patch(color="tab:orange", label="After SA")

            handles.extend([patch1, patch2, line1, line2, line3, line4])


            ax.legend(handles=handles, loc=2, prop={'size':15})
            continue

        if i == 3 and not skip:
            i -= 1
            skip = True


        averages_before = [round(x,3) for x in [dfs[i]["i_EC"].mean(), dfs[i]["i_EO"].mean(), dfs[i]["i_AR"].mean(), dfs[i]["i_EL"].mean(), dfs[i]["i_GR"].mean()]] #bar
        averages_after = [round(x,3) for x in [dfs[i]["f_EC"].mean(), dfs[i]["f_EO"].mean(), dfs[i]["f_AR"].mean(), dfs[i]["f_EL"].mean(), dfs[i]["f_GR"].mean()]]#bar

        both = [averages_before, averages_after]

        chosen_before = round(dfs[i]['Initial Evaluation(Chosen Metrics)'].mean(),3) #line
        chosen_after = round(dfs[i]['Final Evaluation(Chosen Metrics)'].mean(),3) #line

        all_before = round(dfs[i]['Initial Evaluation(All Metrics)'].mean(),3) #dotted line
        all_after = round(dfs[i]['Final Evaluation(All Metrics)'].mean(),3) #dotted line

        labels = ["EC", "EO", "AR", "EL", "GR"]
        y_size = [x / 10 for x in range(11)]

        x = np.arange(len(labels))
        width = 0.35

        before = ax.bar(x - width/2, averages_before, width, label="Before SA")
        after = ax.bar(x + width/2, averages_after, width, label="After SA")

        line_chosen_before = ax.axhline(y=chosen_before, color="blue", linestyle='solid', label="Chosen Metrics before SA")
        line_chosen_after = ax.axhline(y=chosen_after, color="orange", linestyle='solid', label="Chosen Metrics after SA")

        line_all_before = ax.axhline(y=all_before, color="blue", linestyle='dashed', label="All Metrics before SA")
        line_all_after = ax.axhline(y=all_after, color="orange", linestyle='dashed', label="All Metrics after SA")

        ax.set_xticks(x, labels)
        #ax.legend()

        ax.set_yticks(y_size)
        ax.set_ylabel("Metric Value")
        ax.set_xlabel("Metrics")

        #ax.bar_label(before, padding=3)
        #ax.bar_label(after, padding=3)
        
        ax.set_title("Metric Improvements For " + labels[i])
        
        
        i += 1
        

    plt.show()

def compare_before_after3(df):

    EC_df = df[df['EC'] == 1]
    EO_df = df[df['EO'] == 1]
    AR_df = df[df['AR'] == 1]
    EL_df = df[df['EL'] == 1]
    GR_df = df[df['GR'] == 1]

    dfs = [EC_df, EO_df, AR_df, EL_df, GR_df]

    i = 0
    skip = False
    fig, axs = plt.subplots(3,2)
    for ax in axs.flat:
        if i == 1:
            i += 1

            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            handles, labels = plt.gca().get_legend_handles_labels()

            line1 = Line2D([0], [0], color="blue",linewidth=1, linestyle="solid", label="Chosen Metrics before SA")
            line2 = Line2D([0], [0], color="orange",linewidth=1, linestyle="solid", label="Chosen Metrics after SA")
            line3 = Line2D([0], [0], color="blue",linewidth=1, linestyle="dashed", label="All Metrics before SA")
            line4 = Line2D([0], [0], color="orange",linewidth=1, linestyle="dashed", label="All Metrics after SA")

            patch1 = mpatches.Patch(color="tab:blue", label="Before SA")
            patch2 = mpatches.Patch(color="tab:orange", label="After SA")

            handles.extend([patch1, patch2, line1, line2, line3, line4])


            ax.legend(handles=handles, loc=2, prop={'size':15})
            continue

        if i == 2 and not skip:
            i -= 1
            skip = True


        averages_before = [round(x,3) for x in [dfs[i]["i_EC"].mean(), dfs[i]["i_EO"].mean(), dfs[i]["i_AR"].mean(), dfs[i]["i_EL"].mean(), dfs[i]["i_GR"].mean()]] #bar
        averages_after = [round(x,3) for x in [dfs[i]["f_EC"].mean(), dfs[i]["f_EO"].mean(), dfs[i]["f_AR"].mean(), dfs[i]["f_EL"].mean(), dfs[i]["f_GR"].mean()]]#bar

        both = [averages_before, averages_after]

        chosen_before = round(dfs[i]['Initial Evaluation(Chosen Metrics)'].mean(),3) #line
        chosen_after = round(dfs[i]['Final Evaluation(Chosen Metrics)'].mean(),3) #line

        all_before = round(dfs[i]['Initial Evaluation(All Metrics)'].mean(),3) #dotted line
        all_after = round(dfs[i]['Final Evaluation(All Metrics)'].mean(),3) #dotted line

        labels = ["EC", "EO", "AR", "EL", "GR"]
        y_size = [x / 10 for x in range(11)]

        x = np.arange(len(labels))
        width = 0.35

        before = ax.bar(x - width/2, averages_before, width, label="Before SA")
        after = ax.bar(x + width/2, averages_after, width, label="After SA")

        line_chosen_before = ax.axhline(y=chosen_before, color="blue", linestyle='solid', label="Chosen Metrics before SA")
        line_chosen_after = ax.axhline(y=chosen_after, color="orange", linestyle='solid', label="Chosen Metrics after SA")

        line_all_before = ax.axhline(y=all_before, color="blue", linestyle='dashed', label="All Metrics before SA")
        line_all_after = ax.axhline(y=all_after, color="orange", linestyle='dashed', label="All Metrics after SA")

        ax.set_xticks(x, labels)
        #ax.legend()

        ax.set_yticks(y_size)
        ax.set_ylabel("Metric Value")
        ax.set_xlabel("Metrics")

        #ax.bar_label(before, padding=3)
        #ax.bar_label(after, padding=3)
        
        ax.set_title("Metric Improvements For " + labels[i])
        
        
        i += 1

    plt.subplots_adjust(left=0.29,right=0.71,bottom=0.05,top=0.95,wspace=0.179,hspace=0.267)
    plt.show()


def fix_csv_names(filename, modulo):
    """Fixes a bug with incorrect filenames from running the experiments"""
    df = pd.read_csv(filename)

    j = 1
    for i in range(len(df)):
        temp = df.iloc[i]["filename"].split("_")
        temp[0] = "G" + str(j) + "F"
        df.at[i, 'filename'] = "_".join(temp)

        if i % modulo == 0 and i != 0:
            j += 1

    #print(df)
    df.to_csv(filename)



def which_cooling(df):
    
    df_linear = df[df["filename"].str.contains("linear")]

    df_quadratic = df[df["filename"].str.contains("quadratic")]


    print(f"Average improvement for chosen metrics linear: {df_linear['chosen_diff'].mean()}")
    print(f"Average improvement for chosen metrics quadratic: {df_quadratic['chosen_diff'].mean()}")

    print(f"Average improvement for all metrics linear: {df_linear['all_diff'].mean()}")
    print(f"Average improvement for all metrics quadratic: {df_quadratic['all_diff'].mean()}")


def split_data(df):


    df_EC = df[df['EC'] == 1]

    df_EC = df_EC[['filename','Initial Evaluation(All Metrics)', 'Final Evaluation(All Metrics)', 'Initial Evaluation(Chosen Metrics)', 'Final Evaluation(Chosen Metrics)', 
    'chosen_diff','all_diff']]
    print(df_EC)

    df_EC.to_csv("..\\..\\data\\individual\\experiment_1_EC.csv")



def main():
    filename = "..\\..\\data\\experiment_100_1.csv"
    #count_graphs()
    df = load_file(filename)
    #compare_before_after(df)
    #fix_csv_names("..\\..\\data\\experiment_100_0.csv", 40)
    #which_cooling(df)
    split_data(df)

if __name__ == "__main__":
    main()