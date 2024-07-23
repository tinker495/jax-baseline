import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
exit
csvdict = {
    'TD3': 'test/dpg_log/TD3.csv',
    'SAC': 'test/dpg_log/SAC.csv',
    'TQC': 'test/dpg_log/TQC(25)_truncated(5).csv',
    'TD7': 'test/dpg_log/TD7.csv'
}
dfdict = {}
average_dict = {}
maximum_steps = 0
for key in csvdict:
    df = pd.read_csv(csvdict[key])
    df = df.astype(float)
    max_step = df["Step"].max()
    maximum_steps = max(maximum_steps, max_step)
    split_size = (max_step/50)
    df["Split_Step"] = df["Step"].apply(lambda x: round(x/split_size))  # split the "Step" column into groups of 1000
    # split based on "Step" column and get the mean of each group
    #df["Average_Reward"] = df.groupby("Split_Step")["Value"].transform("mean")
    averagedf = df.groupby("Split_Step")["Value"].mean()
    averagedf = averagedf.reset_index()
    averagedf.columns = ["Split_Step", "Average_Reward"]
    averagedf["Step"] = averagedf["Split_Step"] * split_size
    average_dict[key] = averagedf
    dfdict[key] = df

for key in dfdict:
    sns.lineplot(data=average_dict[key], x='Step', y='Average_Reward', label=key)
plt.xlabel('Step')
plt.ylabel('Average Return')
plt.xlim(0, maximum_steps)
plt.title('Average Return')
plt.legend()
plt.savefig('docs/figures/dpg_Humanoid-v4.png')

#get all max average rewards
for key in average_dict:
    print(f"{key}, {average_dict[key]['Average_Reward'].max():.2f}")