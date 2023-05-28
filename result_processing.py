#!/usr/bin/env python3
"""Process output from experiment.py script"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from experiment import preprocs, metrics, clfs, datasets
from tabulate import tabulate
from scipy.stats import ranksums, rankdata, ttest_rel


def get_metric_method_keys():
    """Retrieve key lists from metrics and methods dicts"""
    return list(metrics.keys()), list(preprocs.keys())


def load_scores():
    """Load scores from a file and calc mean of axis 3 (folds)"""
    scores = np.load('results/results.npy')
    scores = np.mean(scores, axis=3)
    return scores


def save_table(filename, table, clf):
    """Save table to a file """
    path = Path('results')
    path.mkdir(parents=True, exist_ok=True)
    with open(path.joinpath(clf + '_' + filename), 'w') as f:
        f.write(table)


def statistic_analysis(ranks, metric_index, clf):
    """Statistic analysis for given metric and ranks matrix.
        Results are saved in results dir in files metric_key.tex and metric_key.txt"""
    metric_keys, method_keys = get_metric_method_keys()
    metric_key = metric_keys[metric_index]
    print('Analyzing for metric: ' + metric_key)
    result_string = ''
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
    result_string += '\n\nw-statistic:\n' + \
        w_statistic_table + '\n\np-value:\n' + p_value_table

    advantage = np.zeros((len(method_keys), len(method_keys)))
    advantage[w_statistic > 0] = 1
    advantage_table = np.concatenate(
        (names_column, advantage), axis=1)
    advantage_table = tabulate(advantage_table, headers)
    result_string += '\n\nAdvantage:\n' + advantage_table

    significance = np.zeros((len(method_keys), len(method_keys)))
    significance[p_value <= alfa] = 1
    significance_table = np.concatenate(
        (names_column, significance), axis=1)

    significance_table = tabulate(significance_table, headers)
    result_string += '\n\nStatistical significance (alpha = 0.05):\n' + \
        significance_table

    stat_better = significance * advantage
    stat_better_table = np.concatenate(
        (names_column, stat_better), axis=1)
    save_table(metric_key + '_stat_better.tex',
               tabulate(stat_better_table, headers, tablefmt='latex', floatfmt='.3f'), clf)
    stat_better_table = tabulate(stat_better_table, headers)
    result_string += '\n\nStatistically significantly better:\n' + stat_better_table

    print(result_string)


def analyze_all_metrics(scores):
    """Perform statistic analysis for all available metrics"""
    metric_keys, _ = get_metric_method_keys()
    for index, _ in enumerate(metric_keys):
        statistic_analysis(scores, index)


def analyze_clf(clf, scores):
    metric_keys, method_keys = get_metric_method_keys()
    plot_data = np.zeros((len(metric_keys), len(method_keys)))
    for metric_id, metric_key in enumerate(metric_keys):
        metric_scores = scores[:, :, metric_id]
        mean_ranks, ranks = calc_ranks(metric_scores)
        if metric_key == 'bac':
            score_table(metric_scores, clf, mean_ranks, ranks)
        plot_data[metric_id] = mean_ranks
    draw_plot(plot_data, metric_keys, method_keys, clf)


def score_table(scores, clf, mean_ranks, ranks):
    t = []
    preprocs_count = len(preprocs)
    for db_idx, db_name in enumerate(datasets):
        # Row with mean scores
        t.append(['%s' % db_name] + ['%.3f' %
                                     v for v in
                                     scores[db_idx, :]])

        alpha = .05
        T = np.zeros((preprocs_count, preprocs_count))
        p = np.zeros((preprocs_count, preprocs_count))

        for i in range(len(clfs)):
            for j in range(len(clfs)):
                T[i, j], p[i, j] = ttest_rel(scores[i], scores[j])

        mean_adv = scores[db_idx, :] < scores[db_idx, :, np.newaxis]
        stat_adv = p < alpha

        _ = np.where(stat_adv * mean_adv)
        conclusions = [list(1 + _[1][_[0] == i])
                       for i in range(preprocs_count)]

        t.append([''] + [", ".join(["%i" % i for i in c])
                         if len(c) > 0 and len(c) < preprocs_count-1 else ("all" if len(c) == preprocs_count-1 else '---')
                         for c in conclusions])

    t.append(['MEAN RANKS'] + list(mean_ranks))
    w_statistic = np.zeros((preprocs_count, preprocs_count))
    p_value = np.zeros((preprocs_count, preprocs_count))
    for i in range(preprocs_count):
        for j in range(preprocs_count):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])
    advantage = np.zeros((preprocs_count, preprocs_count))
    advantage[w_statistic > 0] = 1
    significance = np.zeros((preprocs_count, preprocs_count))
    significance[p_value <= alpha] = 1
    _ = np.where(advantage * significance)
    conclusions = [list(1 + _[1][_[0] == i])
                   for i in range(preprocs_count)]

    t.append([''] + [", ".join(["%i" % i for i in c])
                     if len(c) > 0 and len(c) < preprocs_count-1 else ("all" if len(c) == preprocs_count-1 else '---')
                     for c in conclusions])

    save_table('scores.tex', tabulate(t, headers=[
               'DATASET'] + list(preprocs.keys()), tablefmt='latex', floatfmt='%.3f'), clf)


def calc_ranks(scores):
    ranks = []
    for score in scores:
        ranks.append(rankdata(score).tolist())
    ranks = np.array(ranks)
    mean_ranks = np.mean(ranks, axis=0)
    return mean_ranks, ranks


def draw_plot(plot_data, axis_labels, data_labels, plot_name):
    """Create a radar plot from given data"""
    axis_labels += axis_labels[:1]
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(axis_labels))
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    max_val = 0
    for method_id, method in enumerate(data_labels):
        values = plot_data[:, method_id].tolist()
        values += values[:1]
        if max(values) > max_val:
            max_val = max(values)
        plt.plot(label_loc, values, label=method)

    plt.yticks(np.arange(0, max_val, 0.5),
               color="grey", size=7)
    plt.title(plot_name, size=20, y=1.05)
    plt.thetagrids(np.degrees(label_loc), labels=axis_labels)
    plt.legend(bbox_to_anchor=(0.5, 0), loc="lower center",
               bbox_transform=fig.transFigure, ncol=len(data_labels))
    plt.savefig('results/radar_' + plot_name, dpi=200)


def main():
    scores = load_scores()
    clf_keys = clfs.keys()
    for clf_id, clf in enumerate(clf_keys):
        analyze_clf(clf, scores[clf_id])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
