import pandas as pd





def main():
    # Read file
    file = "..\\..\\data\\experiment_1.csv"
    df = pd.read_csv(file)

    # Drop unused columns
    df = df.drop(columns=['Num Crossings(Initial)', 'Num Crossings(Final)'])

    # Drop CA
    df = df.drop(columns=['CA'])
    df = df.drop(df[df['Final Evaluation(Chosen Metrics)'] == "None"].index)

    # Convert to float
    df['Final Evaluation(Chosen Metrics)'] = pd.to_numeric(df['Final Evaluation(Chosen Metrics)'], downcast='float')
    df['Initial Evaluation(Chosen Metrics)'] = pd.to_numeric(df['Initial Evaluation(Chosen Metrics)'], downcast='float')
    df['Final Evaluation(All Metrics)'] = pd.to_numeric(df['Final Evaluation(All Metrics)'], downcast='float')
    df['Initial Evaluation(All Metrics)'] = pd.to_numeric(df['Initial Evaluation(All Metrics)'], downcast='float')

    # Calculate improvments in metrics
    df['all_diff'] = df['Final Evaluation(All Metrics)'] - df['Initial Evaluation(All Metrics)']
    df['chosen_diff'] = df['Final Evaluation(Chosen Metrics)'] - df['Initial Evaluation(Chosen Metrics)']

    #general(df)
    #which_cooling(df)
    #which_initial_cfg(df)
    #which_metric(df)

def main_2():
    # Read file
    file = "..\\..\\data\\experiment_2.csv"
    df = pd.read_csv(file)

    # Convert to float
    df['Final Evaluation(Chosen Metrics)'] = pd.to_numeric(df['Final Evaluation(Chosen Metrics)'], downcast='float')
    df['Initial Evaluation(Chosen Metrics)'] = pd.to_numeric(df['Initial Evaluation(Chosen Metrics)'], downcast='float')
    df['Final Evaluation(All Metrics)'] = pd.to_numeric(df['Final Evaluation(All Metrics)'], downcast='float')
    df['Initial Evaluation(All Metrics)'] = pd.to_numeric(df['Initial Evaluation(All Metrics)'], downcast='float')

    # Calculate improvments in metrics
    df['all_diff'] = df['Final Evaluation(All Metrics)'] - df['Initial Evaluation(All Metrics)']
    df['chosen_diff'] = df['Final Evaluation(Chosen Metrics)'] - df['Initial Evaluation(Chosen Metrics)']


    #general(df)
    #which_initial_cfg_2(df)

def main_3():
    # Read file
    file = "..\\..\\data\\experiment_100.csv"
    df = pd.read_csv(file)


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

    #general(df)
    #which_metric2(df)
    #which_initial_cfg_2(df)
    metric_improvments(df)

def general(df):

    average_improvement_chosen = df['chosen_diff'].mean()
    print(average_improvement_chosen)

    average_improvement_all = df['all_diff'].mean()
    print(average_improvement_all)

    max_improvement_chosen = df['chosen_diff'].max()
    print(max_improvement_chosen)

    max_improvement_all = df['all_diff'].max()
    print(max_improvement_all)

    min_improvement_chosen = df['chosen_diff'].min()
    print(min_improvement_chosen)

    min_improvement_all = df['all_diff'].min()
    print(min_improvement_all)


def which_cooling(df):
    linear_slice = [True,True,True,True,True,False,False,False,False,False]*4
    quadratic_slice = [False,False,False,False,False,True,True,True,True,True]*4
    linear_df = df.iloc[linear_slice]
    quadratic_df = df.iloc[quadratic_slice]

    # print(linear_df)
    # print(quadratic_df)

    avg_linear_improvement_chosen = linear_df['chosen_diff'].mean()
    print(f"Linear chosen improvement: {avg_linear_improvement_chosen}")

    avg_quadratic_improvement_chosen = quadratic_df['chosen_diff'].mean()
    print(f"Quadratic chosen improvement: {avg_quadratic_improvement_chosen}")

    avg_linear_improvement_all = linear_df['all_diff'].mean()
    print(f"Linear all improvement: {avg_linear_improvement_all}")

    avg_quadratic_improvement_all = quadratic_df['all_diff'].mean()
    print(f"Quadratic all improvement: {avg_quadratic_improvement_all}")
    

def which_initial_cfg(df):
    random_slice = [True]*10 + [False]*30
    grid_slice = [False]*10 + [True]*10 + [False]*20
    poly3_slice = [False]*20 + [True]*10 + [False]*10
    poly5_slice = [False]*30 + [True]*10

    random_df = df.iloc[random_slice]
    grid_df = df.iloc[grid_slice]
    poly3_df = df.iloc[poly3_slice]
    poly5_df = df.iloc[poly5_slice]


    avg_random_improvement_chosen = random_df['chosen_diff'].mean()
    print(f"Random chosen improvement: {avg_random_improvement_chosen}")

    avg_grid_improvement_chosen = grid_df['chosen_diff'].mean()
    print(f"Grid chosen improvement: {avg_grid_improvement_chosen}")

    avg_poly3_improvement_chosen = poly3_df['chosen_diff'].mean()
    print(f"Poly3 chosen improvement: {avg_poly3_improvement_chosen}")

    avg_poly5_improvement_chosen = poly5_df['chosen_diff'].mean()
    print(f"Poly5 chosen improvement: {avg_poly5_improvement_chosen}")

    print()

    avg_random_improvement_all = random_df['all_diff'].mean()
    print(f"Random all improvement: {avg_random_improvement_all}")

    avg_grid_improvement_all = grid_df['all_diff'].mean()
    print(f"Grid all improvement: {avg_grid_improvement_all}")

    avg_poly3_improvement_all = poly3_df['all_diff'].mean()
    print(f"Poly3 all improvement: {avg_poly3_improvement_all}")

    avg_poly5_improvement_all = poly5_df['all_diff'].mean()
    print(f"Poly5 all improvement: {avg_poly5_improvement_all}")

def which_initial_cfg_2(df):
    random_df = df.iloc[::4]
    grid_df = df.iloc[1::4]
    poly3_df = df.iloc[2::4]
    poly5_df = df.iloc[3::4]


    avg_random_improvement_chosen = random_df['chosen_diff'].mean()
    print(f"Random chosen improvement: {avg_random_improvement_chosen}")

    avg_grid_improvement_chosen = grid_df['chosen_diff'].mean()
    print(f"Grid chosen improvement: {avg_grid_improvement_chosen}")

    avg_poly3_improvement_chosen = poly3_df['chosen_diff'].mean()
    print(f"Poly3 chosen improvement: {avg_poly3_improvement_chosen}")

    avg_poly5_improvement_chosen = poly5_df['chosen_diff'].mean()
    print(f"Poly5 chosen improvement: {avg_poly5_improvement_chosen}")

    print()

    avg_random_improvement_all = random_df['all_diff'].mean()
    print(f"Random all improvement: {avg_random_improvement_all}")

    avg_grid_improvement_all = grid_df['all_diff'].mean()
    print(f"Grid all improvement: {avg_grid_improvement_all}")

    avg_poly3_improvement_all = poly3_df['all_diff'].mean()
    print(f"Poly3 all improvement: {avg_poly3_improvement_all}")

    avg_poly5_improvement_all = poly5_df['all_diff'].mean()
    print(f"Poly5 all improvement: {avg_poly5_improvement_all}")

def which_metric(df):
    EC_df = df.iloc[::5]
    EO_df = df.iloc[1::5]
    AR_df = df.iloc[2::5]
    EL_df = df.iloc[3::5]
    GR_df = df.iloc[4::5]

    metric_dfs = {"EC":EC_df, "EO":EO_df, "AR":AR_df, "EL":EL_df, "GR":GR_df}

    for mdf in metric_dfs:
        avg_improvement_chosen = metric_dfs[mdf]['chosen_diff'].mean()
        print(f"Average {mdf} chosen improvement: {avg_improvement_chosen}")

    print()

    for mdf in metric_dfs:
        avg_improvement_all = metric_dfs[mdf]['all_diff'].mean()
        print(f"Average {mdf} all improvement: {avg_improvement_all}")



def which_metric_other_metrics(df, metric, metrics):
    # EC_df = df[df['EC'] == 1]
    # EO_df = df[df['EO'] == 1]
    # AR_df = df[df['AR'] == 1]
    # EL_df = df[df['EL'] == 1]
    # GR_df = df[df['GR'] == 1]

    mdf = df[df[metric] == 1]

    for m in metrics:
        avg_improvement = round(mdf[m+"_diff"].mean(), 3)
        print(f"{metric} improves {m} by {avg_improvement}")


def metric_improvments(df):
    metrics = ["EC", "EO", "AR", "EL", "GR"]
    for m in metrics:
        which_metric_other_metrics(df, m, metrics)
        print()


def which_cfg3(df):
    #df[df['ids'].str.contains("ball")]
    pass


if __name__ == "__main__":
    main_3()