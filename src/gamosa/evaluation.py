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


def compare_before_after(df, name):

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

    #ax2 = ax.twinx()

    before = ax.bar(x - width/2, averages_before, width, label="Before SA", color="tab:green")
    after = ax.bar(x + width/2, averages_after, width, label="After SA", color="tab:red")

    # line_chosen_before = ax.axhline(y=chosen_before, color="blue", linestyle='solid', label="Target Metrics before SA")
    # line_chosen_after = ax.axhline(y=chosen_after, color="orange", linestyle='solid', label="Target Metrics after SA")

    # line_all_before = ax.axhline(y=all_before, color="blue", linestyle='dashed', label="All Metrics before SA")
    # line_all_after = ax.axhline(y=all_after, color="orange", linestyle='dashed', label="All Metrics after SA")

    line_chosen_before = ax.axhline(y=chosen_before, color="green", linestyle='solid', label=f"Target Metrics before SA ({chosen_before:.3f})")
    line_chosen_after = ax.axhline(y=chosen_after, color="red", linestyle='solid', label=f"Target Metrics after SA ({chosen_after:.3f})")

    line_all_before = ax.axhline(y=all_before, color="green", linestyle='dashed', label=f"All Metrics before SA ({all_before:.3f})")
    line_all_after = ax.axhline(y=all_after, color="red", linestyle='dashed', label=f"All Metrics after SA ({chosen_after:.3f})")

    ax.set_xticks(x, labels)
    ax.legend(bbox_to_anchor=(0.5, -0.2), loc='center', ncol=2)

    ax.set_yticks(y_size)
    ax.set_ylabel("Metric Value")
    ax.set_xlabel("Metrics")

    #ax2.set_yticks([chosen_before, chosen_after, all_before, all_after], [chosen_before, chosen_after, all_before, all_after])
    

    # ax.bar_label(before, padding=3)
    # ax.bar_label(after, padding=3)
    
    ax.set_title(f"Target=({','.join(name)}) Improvements")
    plt.savefig(','.join(name) + '.pdf', format='pdf', bbox_inches='tight')

    plt.savefig(','.join(name) +'.png', format='png', bbox_inches='tight')
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

        line_chosen_before = ax.axhline(y=chosen_before, color="blue", linestyle='solid', label="Target Metrics before SA")
        line_chosen_after = ax.axhline(y=chosen_after, color="orange", linestyle='solid', label="Target Metrics after SA")

        line_all_before = ax.axhline(y=all_before, color="blue", linestyle='dashed', label="All Metrics before SA")
        line_all_after = ax.axhline(y=all_after, color="orange", linestyle='dashed', label="All Metrics after SA")

        ax.set_xticks(x, labels)
        #ax.legend()

        ax.set_yticks(y_size)
        ax.set_ylabel("Metric Value")
        ax.set_xlabel("Metrics")

        #ax.bar_label(before, padding=3)
        #ax.bar_label(after, padding=3)
        
        ax.set_title(f"Target=({labels[i]}) Improvements")
        
        
        i += 1

    plt.subplots_adjust(left=0.29,right=0.71,bottom=0.05,top=0.95,wspace=0.179,hspace=0.267)
    plt.show()

def compare_before_after4(df):

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
    print()
    print(f"Average improvement for all metrics linear: {df_linear['all_diff'].mean()}")
    print(f"Average improvement for all metrics quadratic: {df_quadratic['all_diff'].mean()}")


    chosen_avg = [round(x,3) for x in [ df_linear['chosen_diff'].mean(), df_quadratic['chosen_diff'].mean()]] #bar
    all_avg = [round(x,3) for x in [ df_linear['all_diff'].mean(), df_quadratic['all_diff'].mean()]] #bar


    fig, ax = plt.subplots()

    labels = ["linear", "quadratic"]
    y_size = [x / 10 for x in range(11)]

    x = np.arange(len(labels))
    width = 0.35


    chosen = ax.bar(x - width/2, chosen_avg, width, label="Target Metrics Improvement")
    all = ax.bar(x + width/2, all_avg, width, label="All Metrics Improvement")

    ax.set_xticks(x, labels)
    ax.legend(bbox_to_anchor=(0.5, 1.12), loc='center', ncol=2)

    #ax.set_yticks(y_size)
    ax.set_ylabel("Average Improvement")
    ax.set_xlabel("Cooling Schedule")

    ax.set_title(f"Cooling Schedule Improvements (Experiment 1)")
    ax.bar_label(chosen, padding=1)
    ax.bar_label(all, padding=1)
    plt.savefig(f"Cooling Schedule Improvements (Experiment 1)" + '.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f"Cooling Schedule Improvements (Experiment 1)" + '.png', format='png', bbox_inches='tight')

    plt.show()


def which_initial(df, exp_num):
    
    df_random = df[df["filename"].str.contains("random")]
    df_grid = df[df["filename"].str.contains("grid")]
    df_poly3 = df[df["filename"].str.contains("poly3")]
    df_poly5 = df[df["filename"].str.contains("poly5")]

    # print(f"Average improvement for chosen metrics random: {df_random['chosen_diff'].mean()}")
    # print(f"Average improvement for chosen metrics grid: {df_grid['chosen_diff'].mean()}")
    # print(f"Average improvement for chosen metrics poly3: {df_poly3['chosen_diff'].mean()}")
    # print(f"Average improvement for chosen metrics poly5: {df_poly5['chosen_diff'].mean()}")
    # print()
    # print(f"Average improvement for all metrics random: {df_random['all_diff'].mean()}")
    # print(f"Average improvement for all metrics grid: {df_grid['all_diff'].mean()}")
    # print(f"Average improvement for all metrics poly3: {df_poly3['all_diff'].mean()}")
    # print(f"Average improvement for all metrics poly5: {df_poly5['all_diff'].mean()}")

    chosen_avg = [round(x,3) for x in [ df_random['chosen_diff'].mean(), df_grid['chosen_diff'].mean(), df_poly3['chosen_diff'].mean(), df_poly5['chosen_diff'].mean()]] #bar
    all_avg = [round(x,3) for x in [ df_random['all_diff'].mean(), df_grid['all_diff'].mean(), df_poly3['all_diff'].mean(), df_poly5['all_diff'].mean()]] #bar


    fig, ax = plt.subplots()

    labels = ["random", "grid", "poly3", "poly5"]
    y_size = [x / 10 for x in range(11)]

    x = np.arange(len(labels))
    width = 0.35


    chosen = ax.bar(x - width/2, chosen_avg, width, label="Target Metrics Improvement")
    all = ax.bar(x + width/2, all_avg, width, label="All Metrics Improvement")

    ax.set_xticks(x, labels)
    ax.legend(bbox_to_anchor=(0.5, 1.12), loc='center', ncol=2)

    #ax.set_yticks(y_size)
    ax.set_ylabel("Average Improvement")
    ax.set_xlabel("Initial Configuration")

    ax.set_title(f"Initial Configuration Improvements (Experiment {exp_num})")
    ax.bar_label(chosen, padding=1)
    ax.bar_label(all, padding=1)
    plt.savefig(f"Initial Configuration Improvements (Experiment {exp_num})" + '.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f"Initial Configuration Improvements (Experiment {exp_num})" + '.png', format='png', bbox_inches='tight')

    plt.show()


def which_exp_all(df, name):
    print(f"Average improvement for all in ({','.join(name)}) = {df['all_diff'].mean()}")
    return (f"({','.join(name)})", df['all_diff'].mean())

def which_exp_chosen(df, name):
    print(f"Average improvement for target in ({','.join(name)}) = {df['chosen_diff'].mean()}")
    return (f"({','.join(name)})", df['chosen_diff'].mean())


def plot_exp_1(dfs, metrics, exp_no):

    chosen_avg = [round(x['chosen_diff'].mean(),3) for x in dfs]
    all_avg = [round(x['all_diff'].mean(),3) for x in dfs]

    fig, ax = plt.subplots()

    labels = [",\n".join(m) for m in metrics]
    labels = ["(" + s + ")" for s in labels]

    #labels = [f'({",".join(m)})' for m in metrics]
    y_size = [x / 10 for x in range(11)]

    x = np.arange(len(labels))
    width = 0.35


    chosen = ax.bar(x - width/2, chosen_avg, width, label="Target Metrics Improvement")
    all = ax.bar(x + width/2, all_avg, width, label="All Metrics Improvement")

    #ax.set_xticks(x, labels, rotation='vertical')
    ax.set_xticks(x, labels)
    ax.legend(bbox_to_anchor=(0.5, 1.12), loc='center', ncol=2)

    #ax.set_yticks(y_size)
    ax.set_ylabel("Average Improvement")
    ax.set_xlabel("Target Metric(s)")

    ax.set_title(f"Metric Improvements (Experiment {exp_no})")
    ax.bar_label(chosen, padding=1)
    ax.bar_label(all, padding=1)
    plt.savefig(f"Metric Improvements (Experiment {exp_no})" + '.pdf', format='pdf', bbox_inches='tight')
    plt.savefig(f"Metric Improvements (Experiment {exp_no})" + '.png', format='png', bbox_inches='tight')

    plt.show()


def split_data_single(df):
    metrics = ['EC', 'EO', 'AR', 'EL', 'GR']

    for metric in metrics:
        df_single = df[df[metric] == 1]

        df_single = df_single[['filename','Initial Evaluation(All Metrics)', 'Final Evaluation(All Metrics)', 'Initial Evaluation(Chosen Metrics)', 'Final Evaluation(Chosen Metrics)', 
        'chosen_diff','all_diff']]
        #print(df_single)

        df_single.to_csv("..\\..\\data\\individual\\experiment_1_" + metric + ".csv")

def split_data_multi(df):
    single_metrics = ['EC', 'EO', 'AR', 'EL', 'GR']

    metrics = [['EC','AR'],
        ['EC','EO'],
        ['EL','EO'],
        ['EC','GR'],
        ['AR','GR'],
        ['EC','AR','GR'],
        ['EC','EO','EL'],
        ['EC','AR','EL','GR'],
        ['EC','EL'],
        ['EL','GR']
    ]

    
    for metric in metrics:
        not_metric = [m for m in single_metrics if m not in metric]
        #print(not_metric)
        if len(metric) == 3:
            df_single = df[(df[metric[0]] == 1) & (df[metric[1]] == 1) & (df[metric[2]] == 1) & (df[not_metric[0]] == 0) & (df[not_metric[1]] == 0)]
        elif len(metric) == 4:
            df_single = df[(df[metric[0]] == 1) & (df[metric[1]] == 1) & (df[metric[2]] == 1) & (df[metric[3]] == 1) & (df[not_metric[0]] == 0)]
        else:
            df_single = df[(df[metric[0]] == 1) & (df[metric[1]] == 1) & (df[not_metric[0]] == 0) & (df[not_metric[1]] == 0) & (df[not_metric[2]] == 0)]

        df_single = df_single[['filename','Initial Evaluation(All Metrics)', 'Final Evaluation(All Metrics)', 'Initial Evaluation(Chosen Metrics)', 'Final Evaluation(Chosen Metrics)', 
        'chosen_diff','all_diff']]
        #print(df_single)

        df_single.to_csv("..\\..\\data\\individual\\experiment_2_" + "_".join(metric) + ".csv")


def get_distributions_excluding_no_crossings(filename):
    df = pd.read_csv(filename)
    #df = df.drop(columns=['filename', 'SYM', 'CA', 'time'])
    df = df.drop(df[df['filename'].str.contains("hola")].index)
    df = df.drop(columns=['filename', 'SYM', 'time'])

    # Get rid of None valued entries
    df = df.loc[df['CA'] != "None"]
    df["CA"] = pd.to_numeric(df["CA"], downcast="float")

    df = df.drop(df[df['EC'] == 1].index)
    #df = df.drop(columns=['NO'])

    # print(len(df))
    # for col in df:
        
    #     df.hist(column=col, bins=40, figsize=(10,8))
    #     plt.ylim(0,32180)
        
    #     plt.show()
        
        # df.hist(column=col, bins=40)
    

    fig, axs = plt.subplots(ncols=2, nrows=4)
    #fig, axs = plt.subplots()
    #hist = df.hist(bins=20)
    #hist1 = df.hist(bins=20, sharex=True)
    hist2 = df.hist(bins=40, sharex=True, sharey=True, ax=axs)
    

    plt.subplots_adjust(left=0.3,right=0.7,bottom=0.03,top=0.97,wspace=0.25,hspace=0.25)

    plt.show()


def get_distributions(filename):
    df = pd.read_csv(filename)
    #df = df.drop(columns=['filename', 'SYM', 'CA', 'time'])
    df = df.drop(columns=['filename', 'SYM', 'time'])

    # Get rid of None valued entries
    df = df.loc[df['CA'] != "None"]
    df["CA"] = pd.to_numeric(df["CA"], downcast="float")

    # for col in df:
        
    #     df.hist(column=col, bins=40, figsize=(10,8))
    #     plt.ylim(0,32180)
        
    #     plt.show()
        
    #     # df.hist(column=col, bins=40)
    

    fig, axs = plt.subplots(ncols=4, nrows=2)
    #hist = df.hist(bins=20)
    #hist1 = df.hist(bins=20, sharex=True)
    hist2 = df.hist(bins=40, sharex=True, sharey=True, ax=axs)
    

    #plt.subplots_adjust(left=0.3,right=0.7,bottom=0.03,top=0.97,wspace=0.25,hspace=0.25)

    plt.show()

def main():

    # filename = "..\\..\\data\\nathan_distributions.csv"
    # get_distributions(filename)
    

    filename = "..\\..\\data\\experiment_100_0.csv"
    df = load_file(filename)

    #which_cooling(df)

    EC_df = df[df['EC'] == 1]
    EO_df = df[df['EO'] == 1]
    AR_df = df[df['AR'] == 1]
    EL_df = df[df['EL'] == 1]
    GR_df = df[df['GR'] == 1]

    dfs = [EC_df, EO_df, AR_df, EL_df, GR_df]

    metrics = [['EC'], ['EO'], ['AR'], ['EL'], ['GR']]

    #plot_exp_1(dfs, metrics,"1")
    #which_initial(df, "1")

    filename = "..\\..\\data\\experiment_100_1.csv"
    df = load_file(filename)
    #which_initial(df, "2")
    

    #plot_exp_1(dfs, metrics, "2")

    for dfm, name in zip(dfs, metrics):
       #compare_before_after(dfm, name)
       #which_exp(dfm, name)
       pass
    

    #compare_before_after3(df)

    filename = "..\\..\\data\\experiment_100_2.csv"
    df = load_file(filename)
    #which_initial(df, "3")

    print()
    print()

    EC_AR_df = df[(df['EC'] == 1) & (df['AR'] == 1) & (df['GR'] == 0) & (df['EO'] == 0) & (df['EL'] == 0)]
    EC_EO_df = df[(df['EC'] == 1) & (df['EO'] == 1) & (df['GR'] == 0) & (df['EL'] == 0) & (df['AR'] == 0)]
    EL_EO_df = df[(df['EL'] == 1) & (df['EO'] == 1) & (df['EC'] == 0) & (df['GR'] == 0) & (df['AR'] == 0)]
    EC_GR_df = df[(df['EC'] == 1) & (df['GR'] == 1) & (df['EL'] == 0) & (df['EO'] == 0) & (df['AR'] == 0)]
    AR_GR_df = df[(df['AR'] == 1) & (df['GR'] == 1) & (df['EC'] == 0) & (df['EO'] == 0) & (df['EL'] == 0)]

    EC_AR_GR_df = df[(df['EC'] == 1) & (df['AR'] == 1) & (df['GR'] == 1) & (df['EL'] == 0) & (df['EO'] == 0)]
    EC_EO_EL_df = df[(df['EC'] == 1) & (df['EO'] == 1) & (df['EL'] == 1) & (df['GR'] == 0) & (df['AR'] == 0)]
    EC_AR_EL_GR_df = df[(df['EC'] == 1) & (df['AR'] == 1) & (df['EL'] == 1) & (df['GR'] == 1) & (df['EO'] == 0)]

    EC_EL_df = df[(df['EC'] == 1) & (df['EL'] == 1) & (df['GR'] == 0) & (df['EO'] == 0) & (df['AR'] == 0)]
    EL_GR_df = df[(df['EL'] == 1) & (df['GR'] == 1) & (df['EC'] == 0) & (df['EO'] == 0) & (df['AR'] == 0)]

    dfs = [EC_AR_df, EC_EO_df, EL_EO_df, EC_GR_df, AR_GR_df, EC_AR_GR_df, EC_EO_EL_df, EC_AR_EL_GR_df, EC_EL_df, EL_GR_df]

    metrics = [['EC','AR'],
        ['EC','EO'],
        ['EL','EO'],
        ['EC','GR'],
        ['AR','GR'],
        ['EC','AR','GR'],
        ['EC','EO','EL'],
        ['EC','AR','EL','GR'],
        ['EC','EL'],
        ['EL','GR']
    ]

    for dfm, name in zip(dfs, metrics):
        #compare_before_after(dfm, name)
        pass
    
    #plot_exp_1(dfs, metrics, "3")
    
    #compare_before_after3(df)
    #fix_csv_names("..\\..\\data\\experiment_100_0.csv", 40)
    #which_cooling(df)
    
    #split_data_multi(df)
    #which_initial(df)

if __name__ == "__main__":
    main()
    #get_distributions_excluding_no_crossings("..\\..\\data\\nathan_distributions.csv")