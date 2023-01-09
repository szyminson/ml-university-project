#!/usr/bin/env python3
"""Process output from experiment.py script"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from experiment import preprocs, metrics
from tabulate import tabulate
from scipy.stats import ranksums, rankdata

def get_metric_method_keys():
    """Retrieve key lists from metrics and methods dicts"""
    return list(metrics.keys()), list(preprocs.keys())

def load_scores():
    """Load scores from a file and calc mean of axis 3 (folds) and 0 (classifiers)"""
    scores = np.load('results/results.npy')
    scores = np.mean(scores, axis=3)
    scores = np.mean(scores, axis=0)
    return scores

def save_table(filename, table):
    """Save table to a file """
    with open('results/' + filename, 'w') as f:
        f.write(table)

def create_plot(scores):
    """Create a radar plot from scores matrix"""

    # Calc mean of axis 0 (datasets) and transpond the matrix
    # The result is rows->metrics, columns->methods
    plot_data = np.mean(scores, axis=0).T

    # Metric and method labels
    metrics, methods = get_metric_method_keys()

    # Number of metrics
    N = plot_data.shape[0]

    # Angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Spider plot
    ax = plt.subplot(111, polar=True)

    # First axis at the top
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # One axis per one metric
    plt.xticks(angles[:-1], metrics)

    # Axis y
    ax.set_rlabel_position(0)
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    ["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
    color="grey", size=7)
    plt.ylim(0,1)
    # Plots for methods
    for method_id, method in enumerate(methods):
        values=plot_data[:, method_id].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)

    plt.legend(bbox_to_anchor=(1.15, -0.05), ncol=5)
    plt.savefig("results/radar", dpi=200)

def statistic_analysis(scores, metric_index):
    """Statistic analysis for given metric and scores matrix.
        Results are saved in results dir in files metric_key.tex and metric_key.txt"""
    metric_keys, method_keys = get_metric_method_keys()
    metric_key = metric_keys[metric_index]
    metric_scores = scores[:,:,metric_index]
    print('Analyzing for metric: ' + metric_key)
    result_string = ''
    ranks = []
    for ms in metric_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    mean_ranks = np.mean(ranks, axis=0)
    result_string += '\nMethods:\n' + str(method_keys)
    result_string += '\nMean ranks:\n' + str(mean_ranks)
    
    alfa = .05
    w_statistic = np.zeros((len(preprocs), len(preprocs)))
    p_value = np.zeros((len(preprocs), len(preprocs)))

    for i in range(len(preprocs)):
        for j in range(len(preprocs)):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

    headers = method_keys
    names_column = np.expand_dims(np.array(method_keys), axis=1)
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    result_string += '\n\nw-statistic:\n' + w_statistic_table + '\n\np-value:\n' + p_value_table

    advantage = np.zeros((len(method_keys), len(method_keys)))
    advantage[w_statistic > 0] = 1
    advantage_table = np.concatenate(
        (names_column, advantage), axis=1)
    save_table(metric_key + '_advantage.tex', tabulate(advantage_table, headers, tablefmt='latex'))
    advantage_table = tabulate(advantage_table, headers)
    result_string += '\n\nAdvantage:\n' + advantage_table

    significance = np.zeros((len(method_keys), len(method_keys)))
    significance[p_value <= alfa] = 1
    significance_table = np.concatenate(
        (names_column, significance), axis=1)
    save_table(metric_key + '_significance.tex', tabulate(significance_table, headers, tablefmt='latex'))
    significance_table = tabulate(significance_table, headers)
    result_string += '\n\nStatistical significance (alpha = 0.05):\n' + significance_table

    stat_better = significance * advantage
    stat_better_table = np.concatenate(
        (names_column, stat_better), axis=1)
    save_table(metric_key + '_stat_better.tex', tabulate(stat_better_table, headers, tablefmt='latex'))
    stat_better_table = tabulate(stat_better_table, headers)
    result_string += '\n\nStatistically significantly better:\n' + stat_better_table

    save_table(metric_key + '.txt', result_string)
    print(result_string)

def analyze_all_metrics(scores):
    """Perform static analysis for all available metrics"""
    metric_keys, _ = get_metric_method_keys()
    for index, _ in enumerate(metric_keys):
        statistic_analysis(scores, index)

def main():
    scores = load_scores()
    create_plot(scores)
    analyze_all_metrics(scores)
    #statistic_analysis(scores, -2)

    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
    