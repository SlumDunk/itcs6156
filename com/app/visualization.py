import pickle

from com.app.base import cluster_methods
from itcs6156 import settings
import pandas as pd
import matplotlib.pyplot as plt


def read_result(city_name, is_cluster=True):
    """

    :param is_cluster:
    :param city_name:
    :return:
    """
    if is_cluster:
        output_file = settings.DATA_URL + city_name + '/output/result.pkl'
    else:
        output_file = settings.DATA_URL + city_name + '/output/non_cluster_result.pkl'
    with open(output_file, 'rb') as f:
        result = pickle.load(f)
    return result


pass


def clean_cluster_result():
    """

    :return:
    """
    final_df = pd.DataFrame()
    for city_name in cities:
        print(60 * "*")
        print(city_name)
        cluster_result = read_result(city_name, True)
        for cluster_method in cluster_methods:
            result = cluster_result[cluster_method]
            for cluster_label, cluster_value in result.items():
                print(cluster_label)
                for regression_method, regression_value in cluster_value.items():
                    print(60 * "*")
                    if len(regression_value) == 0:
                        continue;
                    array = regression_value[:, [0, 3]]
                    df = pd.DataFrame(array, columns=['sequence', 'mae'])
                    regression_method_list = [regression_method for i in range(30)]
                    df.insert(0, 'regression_method', regression_method_list);

                    cluster_list = [cluster_label for i in range(30)]
                    df.insert(0, 'cluster', cluster_list);

                    city_list = [city_name for i in range(30)]
                    df.insert(0, 'city', city_list);

                    cluster_method_list = [cluster_method for i in range(30)]
                    df.insert(0, 'cluster_method', cluster_method_list);
                    final_df = final_df.append(df)
    print(final_df.info())
    output_file = settings.DATA_URL + '/output/cluster_final_result.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(final_df, f)


def clean_non_cluster_result():
    """

    :return:
    """
    final_df = pd.DataFrame()
    for city_name in cities:
        print(60 * "*")
        print(city_name)
        result = read_result(city_name, False)
        print(result)
        for cluster_label, cluster_value in result.items():
                print(cluster_label)
                for regression_method, regression_value in cluster_value.items():
                    print(60 * "*")
                    if len(regression_value) == 0:
                        continue;
                    df = pd.DataFrame(regression_value, columns=['mae'])
                    sequence_list = [i for i in range(30)]
                    df.insert(0, 'sequence', sequence_list);

                    regression_method_list = [regression_method for i in range(30)]
                    df.insert(0, 'regression_method', regression_method_list);

                    cluster_list = [cluster_label for i in range(30)]
                    df.insert(0, 'cluster', cluster_list);

                    city_list = [city_name for i in range(30)]
                    df.insert(0, 'city', city_list);

                    cluster_method_list = ['-' for i in range(30)]
                    df.insert(0, 'cluster_method', cluster_method_list);
                    final_df = final_df.append(df)
    print(final_df.info())
    print(final_df.head(5))
    output_file = settings.DATA_URL + '/output/non_cluster_final_result.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(final_df, f)


if __name__ == '__main__':
    cities = ['Denver', 'Boston', 'Hawaii', 'Nashville']
    # clean_cluster_result()
    # clean_non_cluster_result()

# %%
f = open(settings.DATA_URL+'output/cluster_final_result.pkl', "rb")
cluster_final_result = pickle.load(f)
f.close()

f = open(settings.DATA_URL+'/output/non_cluster_final_result.pkl', "rb")
non_cluster_final_result = pickle.load(f)
f.close()

# %%
fig, axes = plt.subplots(4, 3, figsize=(15, 15))
for (cm), ax in zip(non_cluster_final_result.groupby(['city', 'regression_method']), axes.flatten()):
    cm[1].plot(x='sequence', y='mae', kind='line', ax=ax, title=f'{cm[0][0]}, {cm[0][1]}')
    ax.get_legend().remove()
    ax.get_xaxis().set_visible(False)
plt.show()

# %%
df_temp = cluster_final_result.loc[(cluster_final_result['cluster_method'] == 'kmeans') &
                                   (cluster_final_result['city'] == 'Hawaii')]
fig, axes = plt.subplots(len(pd.unique(df_temp['regression_method'])),
                         len(pd.unique(df_temp['cluster'])),
                         figsize=(len(pd.unique(df_temp['cluster']))*3, len(pd.unique(df_temp['regression_method']))*3))
for (cm), ax in zip(df_temp.groupby(['regression_method', 'cluster']), axes.flatten()):
    cm[1].plot(x='sequence', y='mae', kind='line', ax=ax, title=f'{cm[0][0]}, {cm[0][1]}')
    ax.get_legend().remove()
    ax.get_xaxis().set_visible(False)
plt.show()

# %%
df_temp_1 = cluster_final_result.loc[(cluster_final_result['sequence'] == 29) & (cluster_final_result['cluster'] == -1)].groupby(['cluster_method',
                                                                                                                                  'city',
                                                                                                                                  'regression_method']).mean().drop(columns=['cluster', 'sequence'])
df_temp_2 = cluster_final_result.loc[(cluster_final_result['sequence'] == 29) & (cluster_final_result['cluster'] > -1)]
df_temp_2 = df_temp_2.groupby(['cluster_method',
                               'city',
                               'regression_method']).mean().drop(columns=['cluster', 'sequence'])
df_temp_1['cluster'] = 'all'
df_temp_2['cluster'] = 'cluster_average'
df_temp = df_temp_1.append(df_temp_2).reset_index()

fig, axes = plt.subplots(4, 6, figsize=(6*3, 4*3))
for (cm), ax in zip(df_temp.reset_index().groupby(['cluster_method',
                                                   'city',
                                                   'regression_method']), axes.flatten()):
    cm[1].plot(x='cluster', y='mae', kind='line', ax=ax, title=f'{cm[0][0]}, {cm[0][1]}, {cm[0][2]}')
    ax.get_legend().remove()
plt.show()












