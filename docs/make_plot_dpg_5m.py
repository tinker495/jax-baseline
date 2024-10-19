import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

matplotlib.use('Agg')

plt.ioff()
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
min_y = 0
max_y = 14000
csvdict = {
    'TD3': 'docs/csv/dpg_humanoid_5m/TD3.csv',
    'SAC': 'docs/csv/dpg_humanoid_5m/SAC.csv',
    'TQC': 'docs/csv/dpg_humanoid_5m/TQC.csv',
    'TD7': 'docs/csv/dpg_humanoid_5m/TD7.csv'
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
    df["Split_Step"] = df["Step"].apply(lambda x: round(x/split_size))
    averagedf = df.groupby("Split_Step")["Value"].mean()
    averagedf = averagedf.reset_index()
    averagedf.columns = ["Split_Step", "Average_Reward"]
    averagedf["Step"] = averagedf["Split_Step"] * split_size
    average_dict[key] = averagedf
    dfdict[key] = df

for key in dfdict:
    sns.lineplot(data=average_dict[key], x='Step', y='Average_Reward', label=key)
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.xlim(0, maximum_steps)
plt.ylim(min_y, max_y)
plt.title('Average Reward')
plt.legend()

#get all max average rewards
for key in average_dict:
    print(f"{key}, {average_dict[key]['Average_Reward'].max():.2f}")

plt.subplot(1, 2, 2)
csvdict = {
    'Simba + TD3': 'docs/csv/dpg_humanoid_5m/Simba_TD3.csv',
    'Simba + SAC': 'docs/csv/dpg_humanoid_5m/Simba_SAC.csv',
    'Simba + TQC': 'docs/csv/dpg_humanoid_5m/Simba_TQC.csv',
    'Simba + TD7': 'docs/csv/dpg_humanoid_5m/Simba_TD7.csv'
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
    df["Split_Step"] = df["Step"].apply(lambda x: round(x/split_size))
    averagedf = df.groupby("Split_Step")["Value"].mean()
    averagedf = averagedf.reset_index()
    averagedf.columns = ["Split_Step", "Average_Reward"]
    averagedf["Step"] = averagedf["Split_Step"] * split_size
    average_dict[key] = averagedf
    dfdict[key] = df

for key in dfdict:
    sns.lineplot(data=average_dict[key], x='Step', y='Average_Reward', label=key)
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.xlim(0, maximum_steps)
plt.ylim(min_y, max_y)
plt.title('Average Reward')
plt.legend()
plt.savefig('docs/figures/dpg_Humanoid-v4-5m.png')

#get all max average rewards
for key in average_dict:
    print(f"{key}, {average_dict[key]['Average_Reward'].max():.2f}")