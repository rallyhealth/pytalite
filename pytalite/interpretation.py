from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import sys
from functools import partial
from matplotlib.gridspec import GridSpec
from multiprocess import Pool

import pytalite.util.color as clr
from pytalite.util.data_ops import df_to_arrays
from pytalite.util.plots import *
from pytalite.plotwrapper import PlotWrapper


# Set path of the mpl style sheet
try:
    import pathlib
    style_path = str(pathlib.Path(__file__).parent.resolve() / 'stylelib/ggplot-transparent.mplstyle')
except ImportError:
    from os import path
    style_path = path.join(path.dirname(path.abspath(__file__)), 'stylelib/ggplot-transparent.mplstyle')


def decile_plot(df, y_column, model, columns_to_exclude=(), num_deciles=10):
    """The function sorts the data points by the predicted positive class probability and divide them into bins.
     It plots bins based on the cumulative precision and recall in two plots.

    Parameters
    ----------
    df : DataFrame
        Data to be plotted

    y_column : str
        Name of the class column

    model : Scikitlearn-like-model
        The model object to be evaluated

    columns_to_exclude : tuple, optional (default=())
        Names of unwanted columns

    num_deciles : int, optional (default=10)
        Number of bars to be plotted, each bar represents about 1/num_deciles of the data

    Returns
    -------
    plot_wrapper : pytalite.plotwrapper.PlotWrapper
        The PlotWrapper object that contains the information and data of the plot

    Raises
    ------
    ValueError
        If the number of deciles exceeds 50
    """

    # Validation check
    if num_deciles > 50:
        raise ValueError("The number of deciles cannot exceed 50")

    # Get X, y array representation of data
    X, y = df_to_arrays(df, y_column, columns_to_exclude)

    # Get and sort predicted probability, then split to 10 arrays (deciles)
    y_prob = model.predict_proba(X)
    indices = np.argsort(y_prob[:, 1])[::-1]
    deciles = list(indices[:indices.shape[0] - indices.shape[0] % num_deciles]
                   .reshape((num_deciles, y_prob.shape[0] // num_deciles)))
    deciles[-1] = np.concatenate((deciles[-1], indices[indices.shape[0] - indices.shape[0] % num_deciles:]))
    true_counts = np.array([np.bincount(y[decile], minlength=2)[1] for decile in deciles])
    decile_size = np.array([decile.shape[0] for decile in deciles])

    # Calculate the true label fraction on each decile and cumulative decile precision
    cum_recall_score = np.cumsum(true_counts) / true_counts.sum()
    cum_precision_score = np.cumsum(true_counts) / np.cumsum(decile_size)

    # Create decile plot
    with plt.style.context(style_path):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 9))
        xticks = np.arange(0, 11) / 10
        xtick_labels = list(map(lambda x: "%d%%" % x, xticks * 100))
        xs = np.arange(num_deciles) / num_deciles

        # Draw bar plot
        bar_plot(ax1, xs, cum_precision_score, width=1 / num_deciles, align='edge',
                 ylim=(0, np.max(cum_precision_score) * 1.2), ylabel="Cumulative precision",
                 edge_color='w', bar_label=False)

        # Create cumulative decile plot
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)

        # Draw bar plot
        bar_plot(ax2, xs, cum_recall_score, width=1 / num_deciles, align='edge',
                 xticks=xticks, xticklabels=xtick_labels, xlim=(0, 1), xlabel="Deciles",
                 ylim=(0, np.max(cum_recall_score) * 1.2), ylabel="Cumulative recall",
                 bar_color=clr.main[0], edge_color='w', bar_label=False)

    plt.show()

    return PlotWrapper(fig, (ax1, ax2), {"shared_x": xs, "cum_recall_score": cum_recall_score,
                                         "cum_precision_score": cum_precision_score})


def _feature_correlation_cat(vals, counts, X, y, model, feature):
    """Compute the proportion of true label and distribution of predicted probability for each category"""
    cat_avg = []
    probs = []

    indices = np.zeros((X.shape[0],), dtype=np.bool)
    for val, count in zip(vals, counts):
        if type(feature) is np.ndarray:
            if val in feature:
                idx = X[:, val] == 1
                indices |= idx

            # The excluded feature values will always be the last element
            else:
                idx = ~indices

            cat_avg.append(y[idx].sum() / count)
            probs.append(model.predict_proba(X[idx])[:, 1])

        else:
            cat_avg.append(y[X[:, feature] == val].sum() / count)
            probs.append(model.predict_proba(X[X[:, feature] == val])[:, 1])
    return np.array(cat_avg), np.array(probs)


def _categorical_fc_plot(X, y, model, features, feature_name, one_hot=False):
    """Create correlation plot for a categorical feature"""
    if one_hot:
        vals = np.append(features, features.max() + 1)
        counts = X[:, features].sum(axis=0)
        counts = np.append(counts, X.shape[0] - counts.sum())
        xticklabels = feature_name + ["others"]
        feature_name = None
    else:
        # Find unique categories, and their corresponding counts
        vals, counts = np.unique(X[:, features], return_counts=True)
        xticklabels = list(map(lambda x: "Cat. %d" % int(x), vals))

    cat_avg, probs = _feature_correlation_cat(vals, counts, X, y, model, features)

    # Create axis labels
    indices = np.arange(len(vals))

    # Plot
    with plt.style.context(style_path):
        fig = plt.figure(figsize=(12, 9))
        grid = GridSpec(3, 1, height_ratios=[5, 5, 0.5], hspace=0.1)
        ax1 = plt.subplot(grid[0])
        fig.add_subplot(ax1)

        # Line part
        line_plot(ax1, indices, cat_avg, marker='o', line_color=clr.main[3],
                  xticks=indices, ylabel="Average", xticklabels=[],
                  xlim=(indices.min() - 0.5, indices.max() + 0.5),
                  ylim=(cat_avg.min() - np.ptp(cat_avg) * 0.2, cat_avg.max() + np.ptp(cat_avg) * 0.2))

        # Violin Part
        ax2 = plt.subplot(grid[1])
        fig.add_subplot(ax2, sharex=ax1)
        violin_plot(ax2, probs, positions=indices, violin_color=clr.main[3], bar_color=clr.main[2], xticks=indices,
                    xticklabels=[],
                    xlim=(indices.min() - 0.5, indices.max() + 0.5), ylabel="Predicted Probability")

        # Distribution plot
        ax3 = plt.subplot(grid[2])
        fig.add_subplot(ax3, sharex=ax1)
        count_plot(ax3, counts, color_map=clr.gen_cmap([(1, 1, 1), clr.main[5], clr.main[6]], [128, 128]),
                   xticklabels=xticklabels, xlabel=feature_name, yticks=[])

    plt.show()
    return PlotWrapper(fig, (ax1, ax2, ax3), {"categories": xticklabels, "pos_label_fraction": cat_avg,
                                              "predicted_probs": probs, "cat_distribution": counts})


def _feature_correlation_num(X, y, feature, n_bins=-1, return_prob=False, model=None):
    """Calculate proportion of true label and distribution predicted probability for a numerical feature using binning
    """
    bin_avg = []
    probs = []

    # Calculating average prob requires model
    if return_prob and model is None:
        raise ValueError("Enabling return_prob requires a model object")

    # Divide bins based on quantiles of unique values (except -1, which means missing data) in the feature space
    unique_feature_vals = np.unique(X[:, feature])
    unique_feature_vals = unique_feature_vals[unique_feature_vals != -1]

    if unique_feature_vals.shape[0] > 10:
        quantiles = np.unique(np.percentile(unique_feature_vals, [i * 100 / n_bins for i in range(n_bins + 1)]))
    else:
        quantiles = np.append(unique_feature_vals, unique_feature_vals.max() + np.diff(unique_feature_vals).min())

    # Calculate correlation (fraction of pos label in actual data)
    for i in range(len(quantiles) - 1):

        # Last interval needs to be inclusive at both boundaries
        if i != len(quantiles) - 2:
            bin_labels = y[(X[:, feature] >= quantiles[i]) & (X[:, feature] < quantiles[i + 1])]
            prob = model.predict_proba(X[(X[:, feature] >= quantiles[i]) & (X[:, feature] < quantiles[i + 1])])
        else:
            bin_labels = y[(X[:, feature] >= quantiles[i]) & (X[:, feature] <= quantiles[i + 1])]
            prob = model.predict_proba(X[(X[:, feature] >= quantiles[i]) & (X[:, feature] <= quantiles[i + 1])])
        bin_avg.append(np.sum(bin_labels) / bin_labels.shape[0])
        probs.append(prob[:, 1])

    bin_avg = np.array(bin_avg)

    return quantiles, bin_avg, probs


def _numerical_fc_plot(X, y, model, feature, feature_name):
    """Create correlation plot for a numerical feature"""

    # Calculate correlation and predicted probability data
    quantiles, bin_avg, probs = _feature_correlation_num(X, y, feature, n_bins=10, return_prob=True, model=model)

    # Create xticklabels of violin plot
    xticklabels = []
    for i in range(len(quantiles) - 1):
        if i != len(quantiles) - 2:
            xticklabels.append("[%.2f,\n%.2f)" % (quantiles[i], quantiles[i + 1]))
        else:
            xticklabels.append("[%.2f,\n%.2f]" % (quantiles[i], quantiles[i + 1]))

    # Parameters used in plot, x-axis is divided evenly for each quantile
    xs = np.arange(len(bin_avg))
    counts = np.array([len(pct_prob) for pct_prob in probs])

    # Plot
    with plt.style.context(style_path):
        fig = plt.figure(figsize=(12, 9))
        grid = GridSpec(3, 1, height_ratios=[5, 5, 0.5], hspace=0.1)
        ax1 = plt.subplot(grid[0])
        fig.add_subplot(ax1)

        # Line part
        line_plot(ax1, xs, bin_avg, marker='o', line_color=clr.main[3],
                  xticks=xs, ylabel="Average", xticklabels=[],
                  xlim=(xs.min() - 0.5, xs.max() + 0.5),
                  ylim=(bin_avg.min() - np.ptp(bin_avg) * 0.2, bin_avg.max() + np.ptp(bin_avg) * 0.2))

        # Violin part
        ax2 = plt.subplot(grid[1])
        fig.add_subplot(ax2, sharex=ax1)
        violin_plot(ax2, probs, positions=xs, violin_color=clr.main[3], bar_color=clr.main[2],
                    xticks=xs, xticklabels=[],
                    xlim=(xs.min() - 0.5, xs.max() + 0.5), ylabel="Predicted Probability")

        # Distribution plot
        ax3 = plt.subplot(grid[2])
        fig.add_subplot(ax3, sharex=ax1)
        count_plot(ax3, counts, color_map=clr.gen_cmap([(1, 1, 1), clr.main[5], clr.main[6]], [128, 128]),
                   xticklabels=xticklabels, xlabel=feature_name, yticks=[])

    plt.show()

    return PlotWrapper(fig, (ax1, ax2, ax3), {"quantiles": quantiles, "pos_label_fraction": bin_avg,
                                              "predicted_probs": probs, "quantile_distribution": counts})


def _feature_type(feature_values):
    """Determine whether a feature is categorical or numerical"""
    unique_values = np.unique(feature_values)

    # Check the length
    if unique_values.shape[0] >= 15:
        return "numerical"

    # Check if each feature values is equally distanced
    dist = np.diff(unique_values)
    if np.all(np.round(dist, 6) == np.round(dist[0], 6)):
        return "categorical"
    else:
        return "numerical"


def feature_correlation_plot(df, y_column, model, feature_column, columns_to_exclude=()):
    """This function detects the feature type and plots the correlation information between feature and actual label.
       The correlation is defined as the fraction of data that has a positive true label (a.k.a. average of true label
       for binary label). It also plots the predicted positive class probability.

    Parameters
    ----------
    df : DataFrame
        Data to be plotted

    y_column : str
        Name of the class column

    model : Scikitlearn-like-model
        The model object to be evaluated

    feature_column : str, or 1d array-like
        Name of the feature column to plot correlation on. If passed in as 1d array-like, the features will be treated
        as one-hot encoded.

    columns_to_exclude : tuple, optional (default=())
        Names of unwanted columns

    Returns
    -------
    plot_wrapper : pytalite.plotwrapper.PlotWrapper
        The PlotWrapper object that contains the information and data of the plot
    """

    # Get X, y array representation and feature indices of data
    X, y, name_to_idx = df_to_arrays(df, y_column, columns_to_exclude, return_index=True)

    if type(feature_column) is str:
        # Set up feature values and feature names
        feature_idx = name_to_idx[feature_column]
        feature_values = X[:, feature_idx]

        # Determine the feature type and plot accordingly
        if _feature_type(feature_values) == "categorical":
            return _categorical_fc_plot(X, y, model, feature_idx, feature_column)
        else:
            return _numerical_fc_plot(X, y, model, feature_idx, feature_column)
    else:  # One-hot features
        feature_idx = np.array([name_to_idx[cat] for cat in feature_column])
        return _categorical_fc_plot(X, y, model, feature_idx, feature_column, one_hot=True)


def density_plot(df, y_column, models, model_names=(), columns_to_exclude=()):
    """This function creates the density plot of predicted positive class probability on actual positive and negative
       data by each model in models in the same plot. It also computes the difference between the distributions on
       positive and negative data using Bhattacharyya distance, KL distance, and cross entropy (a.k.a. log-loss).

    Parameters
    ----------
    df : DataFrame
        Data to be plotted

    y_column : str
        Label of the class column

    models : array-like
        The model objects to be evaluated

    model_names : array-like
        The name of the models to be shown in the legends

    columns_to_exclude : tuple, optional (default=())
        Labels of unwanted columns

    Returns
    -------
    plot_wrapper : pytalite.plotwrapper.PlotWrapper
        The PlotWrapper object that contains the information and data of the plot

    Raises
    ------
    ValueError
        If models is empty or models and model_names does not have the same length
    """

    # Get X, y array representation of data snd predict probability
    X, y = df_to_arrays(df, y_column, columns_to_exclude)
    pos_idx = y == 1
    neg_idx = y == 0
    n_models = len(models)

    if n_models == 0:
        raise ValueError("no models to evaluate")

    if len(model_names) == 0:
        model_names = ["model %d" % (i + 1) for i in range(n_models)]

    if len(model_names) != n_models:
        raise ValueError("models and model_names must have the same length")

    # List and array to store data
    pos_data = np.empty((0, 1000))
    neg_data = np.empty((0, 1000))
    bds = []
    kls = []
    ces = []

    with plt.style.context(style_path):
        fig = plt.figure(figsize=(12, 9))
        grid = GridSpec(2, 1, height_ratios=[3.5, 3.5], hspace=0)
        ax1 = fig.add_subplot(grid[0])
        ax2 = fig.add_subplot(grid[1])
        scores = []

        # Compute density curve for all models
        for model, model_name in zip(models, model_names):
            y_prob = model.predict_proba(X)[:, 1]

            # Fit gaussian kernels on the data
            kernel_pos = st.gaussian_kde(y_prob[pos_idx])
            kernel_neg = st.gaussian_kde(y_prob[neg_idx])

            xs = np.arange(1000) / 1000
            pos_y = kernel_pos(xs)
            neg_y = kernel_neg(xs)

            # Normalize the curve
            pos_norm = (pos_y / pos_y.sum())[np.newaxis, :]
            neg_norm = (neg_y / neg_y.sum())[np.newaxis, :]

            # Compute all three scores
            bd = _bhattacharyya_distance(pos_norm, neg_norm, normalize=True)
            kl = st.entropy(pos_norm[0], neg_norm[0])
            ce = _cross_entropy(pos_norm, neg_norm, normalize=True)

            # Plot using the kernels
            line_plot(ax1, xs, pos_y, legend=model_name, line_color=None, line_label=False)
            line_plot(ax2, xs, neg_y, line_color=None, line_label=False)

            scores.append("%s: Bhattacharyya Distance: %.4f, KL Distance: %.4f, Cross-Entropy: %.4f"
                          % (model_name, bd, kl, ce))

            # Add data
            pos_data = np.vstack((pos_data, pos_y))
            neg_data = np.vstack((neg_data, neg_y))
            bds.append(bd)
            kls.append(kl)
            ces.append(ce)

        ylim_max = max(pos_data.max(), neg_data.max()) * 1.1
        ylim_min = round(-ylim_max * 0.05, 1)

        # Add scores to plot as text
        # ax3.text(0.5, 0.5, "\n".join(scores), va="center", ha="center")

        config_axes(ax1, xticks=[], ylabel="Positive Density", ylim=(ylim_min, ylim_max))
        config_axes(ax2, y_invert=True, xlabel="Probability\n" + "\n".join(scores), ylabel="Negative Density",
                    ylim=(ylim_min, ylim_max))
    plt.show()

    return PlotWrapper(fig, (ax1, ax2), {"probability": xs, "pos_density": pos_data, "neg_density": neg_data,
                                         "Bhattacharyya": np.array(bds), "KL": np.array(kls),
                                         "cross_entropy": np.array(ces)})


def _cross_entropy(y_true, y_prob, normalize=False):
    """Function to calculate cross entropy
       If y_true or y_prob is of shape (num_samples, ),
       the labels are assumed to be binary
    """
    eps = 1e-15

    # Pre-processing
    if y_prob.ndim == 1:
        y_prob = np.vstack((1 - y_prob, y_prob)).T

    if y_true.ndim == 1:
        y_true = np.vstack((1 - y_true, y_true)).T

    y_prob = np.clip(y_prob, eps, 1 - eps)

    # Re-normalize and calculate entropy
    y_prob /= y_prob.sum(axis=1)[:, np.newaxis]

    entropy_arr = -(y_true * np.log(y_prob)).sum(axis=1)

    return entropy_arr.mean() if normalize else entropy_arr.sum()


def _bhattacharyya_distance(y_true, y_prob, normalize=False):
    """Calculates Bhattacharyya distance (https://en.wikipedia.org/wiki/Bhattacharyya_distance)."""

    sim = -np.log(np.sum(np.sqrt(y_true * y_prob), axis=1))
    return sim.mean() if normalize else sim.sum()


def _feature_importance(df, y_column, model, n_jobs, columns_to_exclude=(), n_samples=100):
    """Compute all feature importances by performing multiprocessing"""
    X, y, name_to_idx = df_to_arrays(df, y_column, columns_to_exclude, return_index=True)
    n_jobs = None if n_jobs < 0 else n_jobs

    sample_importance_func = partial(_sample_feature_importance, X=X, y_true=y,
                                     model=model, sample_size=X.shape[0] // n_samples)

    if n_jobs == 1:
        sys.stderr.write("Going single process")
        stats = []
        for stat in map(sample_importance_func, range(n_samples)):
            stats.append(stat)

    else:
        with Pool(n_jobs) as executor_instance:
            chunksize, extra = divmod(n_samples, len(executor_instance._pool))
            if extra:
                chunksize += 1

            stats = []
            sys.stderr.write("Start Multiprocessing, num_processes=%d" % len(executor_instance._pool))
            for stat in executor_instance.map(sample_importance_func, range(n_samples), chunksize):
                stats.append(stat)

    assert len(stats) == 100

    stats = np.array(stats)

    normalized = (stats - np.min(stats)) / np.ptp(stats)

    importance = [(name, _comp_mean_ci(normalized[:, idx])) for name, idx in name_to_idx.items()]
    return sorted(importance, key=lambda x: -x[1][0])


def _sample_feature_importance(_, X, y_true, model, sample_size):
    """Sample the data and compute importance for each feature"""
    sample_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
    X_sample = X[sample_indices]
    y_sample = y_true[sample_indices]
    base_loss = _cross_entropy(y_sample, model.predict_proba(X_sample))

    return [_comp_importance(i, X_sample, y_sample, model, base_loss) for i in range(X_sample.shape[1])]


def _comp_importance(feature, X, y_true, model, base_loss):
    """Compute importance by shuffling the target feature"""
    X_copy = np.copy(X)

    # Permute the feature
    X_copy[:, feature] = np.random.permutation(X_copy[:, feature])
    return _cross_entropy(y_true, model.predict_proba(X_copy)) / base_loss


def _comp_mean_ci(sample, confidence=0.95):
    """Copmute mean and confidence interval from distribution"""
    if np.all(sample == sample[0]):
        return (sample[0],) * 3
    else:
        mean = np.mean(sample)
        ci = st.t.interval(confidence, len(sample), loc=mean, scale=st.sem(sample))
        return mean, ci[0], ci[1]


def feature_importance_plot(df, y_column, model, columns_to_exclude=(), n_jobs=-1, n_top=10):
    """This function computes the importance of each feature to the model by randomly shuffling the feature and compute
       the loss response of the model. It uses log-loss (cross-entropy) as metric.
       Note: This function uses bootstrap.

    Parameters
    ----------
    df : DataFrame
        Data to be plotted

    y_column : str
        Name of the class column

    model : Scikitlearn-like-model
        The model object to be evaluated

    columns_to_exclude : tuple, optional (default=())
        Names of unwanted columns

    n_jobs : int, optional (default=-1)
        Level of multiprocessing, 1 means single-process, -1 means unlimited (actually number of processes depends on
        the machine)

    n_top : int, optional (default=10)
        Number of top features to be plotted

    Returns
    -------
    plot_wrapper : pytalite.plotwrapper.PlotWrapper
        The PlotWrapper object that contains the information and data of the plot

    """
    importance = _feature_importance(df, y_column, model, n_jobs, columns_to_exclude)

    ticks_labels = []
    means = []
    lower_errors = []
    upper_errors = []

    for name, stat in importance[:n_top][::-1]:
        mean, lower_bound, upper_bound = stat
        ticks_labels.append(name)
        means.append(mean)
        lower_errors.append(mean - lower_bound)
        upper_errors.append(upper_bound - mean)

    with plt.style.context(style_path):
        fig, ax = plt.subplots(figsize=(12, 9))
        ticks = np.arange(len(ticks_labels))
        barh_plot(ax, ticks, means, bar_color=clr.main[0], yticks=ticks, yticklabels=ticks_labels, xlim=(0.0, 1.1),
                  xerr=(lower_errors, upper_errors), xlabel='Feature Importance (loss: cross-entropy)')

    plt.show()

    names, stats = zip(*importance)
    return PlotWrapper(fig, (ax,), {"features": names, "stats": np.array(stats)})


def _ale_num(feature, X, predictor, quantiles):
    """Compute ale from quantiles"""
    ale = np.zeros((quantiles.shape[0] - 1,))
    weights = np.zeros((quantiles.shape[0] - 1,), dtype=np.int64)

    for i in range(ale.shape[0]):
        if i != ale.shape[0] - 1:
            quantile_idx = (X[:, feature] >= quantiles[i]) & (X[:, feature] < quantiles[i + 1])
        else:
            quantile_idx = (X[:, feature] >= quantiles[i]) & (X[:, feature] <= quantiles[i + 1])
        X_quantile = X[quantile_idx]

        # ignore zero entries, which should have ale of zero
        if X_quantile.shape[0] != 0:
            weights[i] = X_quantile.shape[0]

            lower_bound = np.copy(X_quantile)
            upper_bound = np.copy(X_quantile)

            lower_bound[:, feature] = quantiles[i]
            upper_bound[:, feature] = quantiles[i + 1]

            ale[i] = np.mean(predictor(upper_bound) - predictor(lower_bound))

    ale = ale.cumsum()

    return ale - np.average(ale, weights=weights), weights


def feature_ale_plot(df, y_column, model, feature_column, predictor=None, columns_to_exclude=(), bins=100):
    """This function create the Accumulated Local Effect (ALE) plot of the target feature.
       Visit https://christophm.github.io/interpretable-ml-book/ale.html for more detailed explanation of ALE.

    Parameters
    ----------
    df : DataFrame
        Data to be plotted

    y_column : str
        Name of the class column

    model : Scikitlearn-like-model
        The model object to be evaluated

    feature_column : str
        Name of the feature column to plot ALE on

    predictor : function, optional (default=None)
        The prediction function, which should take in the feature matrix and return an array of predictions
        The function should output positive class probabilities for a classification task, and actual predicted values
        for a regression task.
        If not specified, defaults to a function equivalent to: lambda X: model.predict_prob(X)[:, 1], which is for
        classification.

    columns_to_exclude : tuple, optional (default=())
        Names of unwanted columns

    bins : int, optional (default=100)
        The number of intervals for the ALE plot

    Returns
    -------
    plot_wrapper : pytalite.plotwrapper.PlotWrapper
        The PlotWrapper object that contains the information and data of the plot
    """
    # Get X, y array representation and feature indices from data
    X, _, name_to_idx = df_to_arrays(df, y_column, columns_to_exclude, return_index=True)
    feature_idx = name_to_idx[feature_column]

    if predictor is None:
        def predictor(X):
            return model.predict_proba(X)[:, 1]

    unique_feature_vals = np.unique(X[:, feature_idx])
    unique_feature_vals = unique_feature_vals[unique_feature_vals != -1]

    quantiles = np.percentile(unique_feature_vals, [i * 100 / bins for i in range(0, bins + 1)])
    ale, counts = _ale_num(feature_idx, X, predictor, quantiles)

    xs = (quantiles[1:] + quantiles[:-1]) / 2

    with plt.style.context(style_path):
        fig = plt.figure(figsize=(12, 9))
        grid = GridSpec(2, 1, height_ratios=[10, 1], hspace=0)

        ax1 = plt.subplot(grid[0])
        fig.add_subplot(ax1)
        line_plot(ax1, xs, ale, line_label=False, xticks=[], ylabel="ALE of %s" % y_column)

        ax2 = plt.subplot(grid[1])
        fig.add_subplot(ax2, sharex=ax1)
        event_plot(ax2, X[:, feature_idx][X[:, feature_idx] != -1], 0.5, 1,
                   xlabel=feature_column, yticks=[], ylim=(-0.2, 1.2))
    plt.show()

    return PlotWrapper(fig, (ax1, ax2), {"quantiles": quantiles, "ale": ale, "quantile_distribution": counts})


def _comp_pd_with_value(val, X, predictor, feature):
    """Compute PD value with feature value replaced by val"""
    X_copy = np.copy(X)

    if type(feature) is np.ndarray:
        X_copy[:, feature[feature == val]] = 1
        X_copy[:, feature[feature != val]] = 0
    else:
        X_copy[:, feature] = val

    return predictor(X_copy).mean()


def _partial_dependence(vals, X, predictor, feature, n_jobs):
    """Compute PD values for each value in vals using multiprocessing"""
    n_jobs = None if n_jobs < 0 else n_jobs

    partial_dependence_func = partial(_comp_pd_with_value, X=X, feature=feature,
                                      predictor=predictor)

    if n_jobs == 1:
        sys.stderr.write("Going single process")
        stats = []
        for stat in map(partial_dependence_func, vals):
            stats.append(stat)

    else:
        with Pool(n_jobs) as executor_instance:
            chunksize, extra = divmod(len(vals), len(executor_instance._pool))
            if extra:
                chunksize += 1

            stats = []
            sys.stderr.write("Start Multiprocessing, num_processes=%d" % len(executor_instance._pool))
            for stat in executor_instance.map(partial_dependence_func, vals, chunksize):
                stats.append(stat)

    return np.array(stats)


def _partial_dependence_plot_cat(X, predictor, features, y_name, feature_name, n_jobs, one_hot=False):
    """Create PDP for categorical feature"""
    if one_hot:
        vals = np.append(features, features.max() + 1)
        counts = X[:, features].sum(axis=0)
        counts = np.append(counts, X.shape[0] - counts.sum())
        xticklabels = feature_name + ["others"]
        feature_name = None
    else:
        vals, counts = np.unique(X[:, features], return_counts=True)
        xticklabels = list(map(lambda x: "Cat. %d" % int(x), vals))

    stats = _partial_dependence(vals, X, predictor, features, n_jobs)

    indices = np.arange(len(vals))

    with plt.style.context(style_path):
        fig = plt.figure(figsize=(12, 9))
        grid = GridSpec(2, 1, height_ratios=[7, 0.5], hspace=0.1)
        ax1 = plt.subplot(grid[0])
        fig.add_subplot(ax1)

        line_plot(ax1, indices, stats, marker='o', xticks=indices, xticklabels=[],
                  ylabel="Mean response of %s" % y_name, xlim=(indices.min() - 0.5, indices.max() + 0.5),
                  ylim=(stats.min() - np.ptp(stats) * 0.2, stats.max() + np.ptp(stats) * 0.2))

        ax2 = plt.subplot(grid[1])
        fig.add_subplot(ax1, sharex=ax1)
        count_plot(ax2, counts, xticklabels=xticklabels, xlabel=feature_name, yticks=[])

    plt.show()

    return PlotWrapper(fig, (ax1, ax2), {"categories": xticklabels, "pd": stats, "cat_distribution": counts})


def _partial_dependence_plot_num(X, predictor, feature, y_name, feature_name, n_jobs):
    """Create PDP for numerical feature"""
    bins = 100
    unique_feature_vals = np.unique(X[:, feature])
    unique_feature_vals = unique_feature_vals[unique_feature_vals != -1]
    quantiles = np.percentile(unique_feature_vals, [i * 100 / bins for i in range(0, bins + 1)])
    xs = (quantiles[1:] + quantiles[:-1]) / 2
    counts = []

    for i in range(len(quantiles) - 1):

        # Last interval needs to be inclusive at both boundaries
        if i != len(quantiles) - 2:
            count = X[(X[:, feature] >= quantiles[i]) & (X[:, feature] < quantiles[i + 1])].shape[0]
        else:
            count = X[(X[:, feature] >= quantiles[i]) & (X[:, feature] <= quantiles[i + 1])].shape[0]
        counts.append(count)

    counts = np.array(counts)
    stats = _partial_dependence(xs, X, predictor, feature, n_jobs)

    with plt.style.context(style_path):
        fig = plt.figure(figsize=(12, 9), facecolor=(1, 1, 1, 0))
        grid = GridSpec(2, 1, height_ratios=[10, 1], hspace=0)

        ax1 = plt.subplot(grid[0])
        fig.add_subplot(ax1)
        line_plot(ax1, xs, stats, line_label=False, xticks=[], ylabel="Mean response of %s" % y_name)

        ax2 = plt.subplot(grid[1])
        fig.add_subplot(ax2, sharex=ax1)
        event_plot(ax2, X[:, feature][X[:, feature] != -1], 0.5, 1,
                   xlabel=feature_name, yticks=[], ylim=(-0.2, 1.2))

    plt.show()

    return PlotWrapper(fig, (ax1, ax2), {"quantiles": quantiles, "pd": stats, "quantile_distribution": counts})


def partial_dependence_plot(df, y_column, model, feature_column, predictor=None, columns_to_exclude=(), n_jobs=-1):
    """This function create the Partial Dependence plot (PDP) of the target feature.
       Visit https://christophm.github.io/interpretable-ml-book/pdp.html for more detailed explanation of PDP.

    Parameters
    ----------
    df : DataFrame
        Data to be plotted

    y_column : str
        Name of the class column

    model : Scikitlearn-like-model
        The model object to be evaluated

    feature_column : str, or 1d array-like
        Name of the feature column to plot PDP on. If passed in as 1d array-like, the features will be treated as one-
        hot encoded.

    predictor : function, optional (default=None)
        The prediction function, which should take in the feature matrix and return an array of predictions
        The function should output positive class probabilities for a classification task, and actual predicted values
        for a regression task.
        If not specified, defaults to a function equivalent to: lambda X: model.predict_prob(X)[:, 1], which is for
        classification.

    columns_to_exclude : tuple, optional (default=())
        Names of unwanted columns

    n_jobs : int, optional (default=-1)
        Level of multiprocessing, 1 means single-process, -1 means unlimited (actually number of processes depends on
        the machine)

    Returns
    -------
    plot_wrapper : pytalite.plotwrapper.PlotWrapper
        The PlotWrapper object that contains the information and data of the plot
    """
    X, _, name_to_idx = df_to_arrays(df, y_column, columns_to_exclude, return_index=True)

    if predictor is None:
        def predictor(X):
            return model.predict_proba(X)[:, 1]

    if type(feature_column) is str:
        feature_idx = name_to_idx[feature_column]
        feature_values = X[:, feature_idx]

        if _feature_type(feature_values) == "categorical":
            return _partial_dependence_plot_cat(X, predictor, feature_idx, y_column, feature_column, n_jobs)
        else:
            return _partial_dependence_plot_num(X, predictor, feature_idx, y_column, feature_column, n_jobs)
    else:
        feature_idx = np.array([name_to_idx[cat] for cat in feature_column])
        return _partial_dependence_plot_cat(X, predictor, feature_idx, y_column, feature_column, n_jobs, one_hot=True)
