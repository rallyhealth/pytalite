from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from matplotlib.gridspec import GridSpec

import pyspark.sql.functions as F
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.sql.window import Window

import pytalite_spark.util.color as clr
from pytalite_spark.util.data_ops import *
from pytalite_spark.util.plots import *
from pytalite_spark.plotwrapper import PlotWrapper

import matplotlib as mpl
if mpl.__version__ <= '1.4.3':
    style_path = 'stylelib/ggplot-transparent-old.mplstyle'
else:
    style_path = 'stylelib/ggplot-transparent.mplstyle'
del mpl

# Set path of the mpl style sheet
try:
    import pathlib
    style_path = str(pathlib.Path(__file__).parent.resolve() / style_path)
except ImportError:
    from os import path
    style_path = path.join(path.dirname(path.abspath(__file__)), style_path)

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()


def decile_plot(df, y_column, model, model_input_col='features', columns_to_exclude=(), num_deciles=10):
    """The function sorts the data points by the predicted positive class probability and divide them into bins.
     It plots bins based on the cumulative precision and recall in two plots.

    Parameters
    ----------
    df : DataFrame
        Data to be plotted

    y_column : str
        Name of the class column

    model : pyspark.ml model
        The model object to be evaluated

    model_input_col : str, optional (default='features')
        The name of the input column of the model, this is also the name of the output column of the VectorAssembler
        that creates the feature vector

    columns_to_exclude : tuple, optional (default=())
        Names of unwanted columns

    num_deciles : int, optional (default=10)
        Number of bars to be plotted, each bar represents about 1/num_deciles of the data

    Returns
    -------
    plot_wrapper : pytalite_spark.plotwrapper.PlotWrapper
        The PlotWrapper object that contains the information and data of the plot

    Raises
    ------
    ValueError
        If the number of deciles exceeds 50
    """

    # Validation check
    if num_deciles > 50:
        raise ValueError("The number of deciles cannot exceed 50")

    # Get preprocessed df and create vector assembler
    df, assembler = preprocess(df, y_column, model_input_col, columns_to_exclude)

    # Predict probability
    pred = model.transform(assembler.transform(df))
    prob_label = pred.select(p1_proba('probability').alias('p1'), F.col(y_column))

    # Label each row with appropriate deciles
    decile_window = Window.partitionBy().orderBy(prob_label.p1)
    decile_prob_label = prob_label.select(F.ntile(num_deciles).over(decile_window).alias("decile"), 'p1', y_column)
    decile_prob_label.cache()

    # Calculate decile size and true counts
    decile_stats = decile_prob_label.groupBy('decile') \
                                    .agg(F.count(y_column).alias('size'), F.sum(y_column).alias('true_count')) \
                                    .crossJoin(decile_prob_label.select(F.sum(y_column).alias('true_sum')))

    cum_window = Window.orderBy(decile_stats.decile.desc()).rangeBetween(Window.unboundedPreceding, 0)

    # Calculate decile scores
    scores = decile_stats.select(F.sum('true_count').over(cum_window).alias('cum_count'),
                                 F.sum('size').over(cum_window).alias('cum_size'), F.col('true_sum'))

    scores = scores.select(F.col('cum_count') / F.col('cum_size'), F.col('cum_count') / F.col('true_sum'))\
                   .toPandas().values

    cum_decile_precision = scores[:, 0]
    cum_decile_recall = scores[:, 1]

    # Create decile plot
    with plt.style.context(style_path):
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 9))
        xticks = np.arange(0, 11) / 10
        xtick_labels = list(map(lambda x: "%d%%" % x, xticks * 100))
        xs = np.arange(num_deciles) / num_deciles

        # Draw bar plot
        bar_plot(ax1, xs, cum_decile_precision, width=1 / num_deciles, align='edge',
                 ylim=(0, np.max(cum_decile_precision) * 1.2), ylabel="True Label Fraction",
                 edge_color='w', bar_label=False)

        # Create cumulative decile plot
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)

        # Draw bar plot
        bar_plot(ax2, xs, cum_decile_recall, width=1 / num_deciles, align='edge',
                 xticks=xticks, xticklabels=xtick_labels, xlim=(0, 1), xlabel="Deciles",
                 ylim=(0, np.max(cum_decile_recall) * 1.2), ylabel="Cumulative True Label Fraction",
                 bar_color=clr.main[0], edge_color='w', bar_label=False)

    plt.show()

    return PlotWrapper(fig, (ax1, ax2), {"shared_x": xs, "cum_precision_score": cum_decile_precision,
                                         "cum_recall_score": cum_decile_recall})


def _feature_correlation_cat(df, y_column, model, feature_column, assembler):
    """Compute the proportion of true label and distribution of predicted probability for each category"""
    pred = model.transform(assembler.transform(df)).select(F.col(feature_column),
                                                           p1_proba('probability').alias('p1'),
                                                           F.col(y_column))
    stats = pred.groupBy(feature_column)\
                .agg(F.count(F.col(feature_column)),
                     (F.sum(y_column) / F.count(F.col(feature_column))).alias('pos_frac'),
                     F.collect_list('p1')).orderBy(feature_column)

    vals, counts, cat_avg, probs = stats.toPandas().values.T
    probs = np.array(list(map(np.array, probs)))
    return vals, counts.astype(np.int64), cat_avg.astype(np.float64), probs


def _categorical_fc_plot(df, y_column, model, feature_column, assembler):
    """Create correlation plot for a categorical feature"""

    vals, counts, cat_avg, probs = _feature_correlation_cat(df, y_column, model, feature_column, assembler)
    try:
        vals = vals.astype(np.float64)
        xticklabels = list(map(lambda x: "Cat. %d" % int(x), vals))
    except ValueError:
        xticklabels = vals

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
                   xticklabels=xticklabels, xlabel=feature_column, yticks=[])

    plt.show()
    return PlotWrapper(fig, (ax1, ax2, ax3), {"categories": xticklabels, "pos_label_fraction": cat_avg,
                                              "predicted_probs": probs, "cat_distribution": counts})


def _feature_correlation_num(df, y_column, model, feature_column, assembler):
    """Calculate proportion of true label and distribution predicted probability for a numerical feature using binning
    """
    n_bins = 12
    feature_vals = df.select(feature_column).filter(F.col(feature_column) != -1)
    unique_feature_vals = np.array(feature_vals.distinct()
                                   .orderBy(feature_column).rdd.map(lambda row: row[0]).collect())

    if unique_feature_vals.shape[0] > 12:
        quantiles = np.unique(np.percentile(unique_feature_vals, [i * 100 / n_bins for i in range(n_bins + 1)]))
    else:
        quantiles = np.append(unique_feature_vals, unique_feature_vals.max() + np.diff(unique_feature_vals).min())

    quantiles = quantiles.astype(np.float64).tolist()

    bins = spark.createDataFrame(list(zip(quantiles[:-1], quantiles[1:], list(range(len(quantiles) - 1)))),
                                 schema=StructType([StructField("lower", DoubleType()),
                                                    StructField("upper", DoubleType()),
                                                    StructField("interval", IntegerType())]))

    df = df.join(bins, ((df[feature_column] >= bins.lower) & (df[feature_column] < bins.upper)), how='left')
    df = df.fillna(quantiles[-2], subset='lower') \
           .fillna(quantiles[-1], subset='upper') \
           .fillna(len(quantiles) - 2, subset='interval')

    pred = model.transform(assembler.transform(df)).select(F.col('interval'),
                                                           p1_proba('probability').alias('p1'),
                                                           F.col(y_column))
    stats = pred.groupBy('interval') \
                .agg(F.count(F.col('interval')),
                     (F.sum(y_column) / F.count(F.col('interval'))).alias('pos_frac'),
                     F.collect_list('p1')).orderBy('interval')

    intervals, counts, cat_avg, probs = stats.toPandas().values.T
    interval_range = list(zip(quantiles[:-1], quantiles[1:]))
    quantiles = [quantiles[0]] + [interval_range[i][1] for i in sorted(intervals)]
    probs = np.array(list(map(np.array, probs)))

    return quantiles, counts.astype(np.int64), cat_avg.astype(np.float64), probs


def _numerical_fc_plot(df, y_column, model, feature_column, assembler):
    """Create correlation plot for a numerical feature"""

    # Calculate correlation and predicted probability data
    quantiles, counts, bin_avg, probs = _feature_correlation_num(df, y_column, model, feature_column, assembler)
    assert counts.sum() == df.count()
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
                   xticklabels=xticklabels, xlabel=feature_column, yticks=[])

    plt.show()

    return PlotWrapper(fig, (ax1, ax2, ax3), {"quantiles": quantiles, "pos_label_fraction": bin_avg,
                                              "predicted_probs": probs, "quantile_distribution": counts})


def _feature_type(df, feature_column):
    """Determine whether a feature is categorical or numerical"""
    unique_values = df.select(feature_column).distinct()
    unique_values.cache()

    # Check the length
    if unique_values.count() >= 15:
        return "numerical"

    # Check if each feature values is equally distanced
    diff_window = Window.partitionBy().orderBy(feature_column)
    unique_values = unique_values.withColumn("prev", (F.lag(feature_column, 1)).over(diff_window)).dropna()
    dist = unique_values.select(F.round(F.col(feature_column) - F.col('prev'), 6).alias('diff'))
    if dist.distinct().count() == 1:
        return "categorical"
    else:
        return "numerical"


def feature_correlation_plot(df, y_column, model, feature_column, model_input_col='features', columns_to_exclude=()):
    """This function detects the feature type and plots the correlation information between feature and actual label.
       The correlation is defined as the fraction of data that has a positive true label (a.k.a. average of true label
       for binary label). It also plots the predicted positive class probability.

    Parameters
    ----------
    df : DataFrame
        Data to be plotted

    y_column : str
        Name of the class column

    model : pyspark.ml model
        The model object to be evaluated

    feature_column : str, or 1d array-like
        Name of the feature column to plot correlation on.

    model_input_col : str, optional (default='features')
        The name of the input column of the model, this is also the name of the output column of the VectorAssembler
        that creates the feature vector

    columns_to_exclude : tuple, optional (default=())
        Names of unwanted columns

    Returns
    -------
    plot_wrapper : pytalite_spark.plotwrapper.PlotWrapper
        The PlotWrapper object that contains the information and data of the plot
    """

    # Get X, y array representation and feature indices of data
    df, assembler = preprocess(df, y_column, model_input_col, columns_to_exclude)

    if type(feature_column) is str:
        # Determine the feature type and plot accordingly
        if _feature_type(df, feature_column) == "categorical":
            return _categorical_fc_plot(df, y_column, model, feature_column, assembler)
        else:
            return _numerical_fc_plot(df, y_column, model, feature_column, assembler)
    else:  # One-hot features
        # feature_idx = np.array([name_to_idx[cat] for cat in feature_column])
        # return _categorical_fc_plot(X, y, model, feature_idx, feature_column, one_hot=True)
        pass


def density_plot(df, y_column, models, model_names=(), model_input_col="features", columns_to_exclude=()):
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

    model_input_col : str, optional (default='features')
        The name of the input column of the models, this is also the name of the output column of the VectorAssembler
        that creates the feature vector. This is expected to be consistent across all models.

    columns_to_exclude : tuple, optional (default=())
        Labels of unwanted columns. This is expected to be consistent across all models.

    Returns
    -------
    plot_wrapper : pytalite_spark.plotwrapper.PlotWrapper
        The PlotWrapper object that contains the information and data of the plot

    Raises
    ------
    ValueError
        If models is empty or models and model_names does not have the same length
    """

    # Get X, y array representation of data snd predict probability
    df, assembler = preprocess(df, y_column, model_input_col, columns_to_exclude)
    pos_df = df.filter(F.col(y_column) == 1)
    pos_df.cache()
    neg_df = df.filter(F.col(y_column) == 0)
    neg_df.cache()
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
        fig = plt.figure(figsize=(12, 9), facecolor=(1, 1, 1, 0))
        grid = GridSpec(2, 1, height_ratios=[3.5, 3.5], hspace=0)
        ax1 = fig.add_subplot(grid[0])
        ax2 = fig.add_subplot(grid[1])
        scores = []

        # Compute density curve for all models
        for model, model_name in zip(models, model_names):
            # TODO Might be better to encapsulate this part
            pos_prob = model.transform(assembler.transform(pos_df)).select(p1_proba('probability')).toPandas().values
            neg_prob = model.transform(assembler.transform(neg_df)).select(p1_proba('probability')).toPandas().values

            # Fit gaussian kernels on the data
            kernel_pos = st.gaussian_kde(pos_prob.flatten())
            kernel_neg = st.gaussian_kde(neg_prob.flatten())

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

        config_axes(ax1, xticks=[], ylabel="Positive Density", ylim=(ylim_min, ylim_max))
        config_axes(ax2, y_invert=True, xlabel="Probability\n" + "\n".join(scores), ylabel="Negative Density",
                    ylim=(ylim_min, ylim_max))
    plt.show()

    return PlotWrapper(fig, (ax1, ax2), {"probability": xs, "pos_density": pos_data, "neg_density": neg_data,
                                         "Bhattacharyya": np.array(bds), "KL": np.array(kls),
                                         "cross_entropy": np.array(ces)})


def _cross_entropy(y_true, y_prob, df=None, normalize=False):
    """Function to calculate cross entropy
       If y_true or y_prob is of shape (num_samples, ),
       the labels are assumed to be binary
    """
    eps = 1e-15

    if df is None:
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

    else:
        df = df.withColumn(y_prob, F.when(F.col(y_prob) < eps, eps)
                                    .when(F.col(y_prob) > (1 - eps), 1 - eps)
                                    .otherwise(F.col(y_prob)))
        df = df.withColumn('entropy', -F.col(y_true) * F.log(F.col(y_prob)) -
                           (1 - F.col(y_true)) * F.log(1 - F.col(y_prob)))

        if normalize:
            return df.agg(F.avg('entropy').alias('loss')).select('loss')
        else:
            return df.agg(F.sum('entropy').alias('loss')).select('loss')


def _bhattacharyya_distance(y_true, y_prob, normalize=False):
    """Calculates Bhattacharyya distance (https://en.wikipedia.org/wiki/Bhattacharyya_distance)."""

    sim = -np.log(np.sum(np.sqrt(y_true * y_prob), axis=1))
    return sim.mean() if normalize else sim.sum()


def _feature_importance(df, y_column, model, model_input_col, columns_to_exclude):
    """Function to calculate importance of all features"""
    df, assembler = preprocess(df, y_column, model_input_col, columns_to_exclude, create_id=True)
    loss_df = spark.createDataFrame(sc.emptyRDD(), schema=StructType([StructField('loss', DoubleType())]))
    pred = model.transform(assembler.transform(df)).select(p1_proba('probability').alias('p1'), y_column)
    base_loss = _cross_entropy(y_column, 'p1', pred).withColumnRenamed('loss', 'base_loss')
    features = []
    input_cols = set(assembler.getInputCols())

    for col in df.columns:
        if col in input_cols:
            features.append(col)
            loss_df = loss_df.union(_comp_feature_loss(df, y_column, model, col, assembler))

    importance_df = loss_df.crossJoin(base_loss).select(F.col('loss') / F.col('base_loss'))
    importance = np.array(importance_df.rdd.map(lambda row: row[0]).collect())
    importance = (importance - np.min(importance)) / np.ptp(importance)

    return sorted(list(zip(features, importance)), key=lambda x: -x[1])


def _comp_feature_loss(df, y_column, model, feature_column, assembler):
    """Compute importance by shuffling the target feature"""
    permuted = add_column_index(df.select(feature_column).orderBy(F.rand()))
    df = df.drop(feature_column).join(permuted, on='_id_')

    pred = model.transform(assembler.transform(df)).select(p1_proba('probability').alias('p1'), F.col(y_column))
    return _cross_entropy(y_column, 'p1', pred)


def feature_importance_plot(df, y_column, model, model_input_col='features', columns_to_exclude=(), n_top=10):
    """This function computes the importance of each feature to the model by randomly shuffling the feature and compute
       the loss response of the model. It uses log-loss (cross-entropy) as metric.

    Parameters
    ----------
    df : DataFrame
        Data to be plotted

    y_column : str
        Name of the class column

    model : pyspark.ml model
        The model object to be evaluated

    model_input_col : str, optional (default='features')
        The name of the input column of the model, this is also the name of the output column of the VectorAssembler
        that creates the feature vector

    columns_to_exclude : tuple, optional (default=())
        Names of unwanted columns

    n_top : int, optional (default=10)
        Number of top features to be plotted

    Returns
    -------
    plot_wrapper : pytalite_spark.plotwrapper.PlotWrapper
        The PlotWrapper object that contains the information and data of the plot
    """
    importance = _feature_importance(df, y_column, model, model_input_col, columns_to_exclude)

    ticks_labels, means = zip(*importance)
    top_ticks_labels = ticks_labels[:n_top][::-1]
    top_means = means[:n_top][::-1]

    with plt.style.context(style_path):
        fig, ax = plt.subplots(figsize=(12, 9))
        ticks = np.arange(len(top_ticks_labels))
        barh_plot(ax, ticks, top_means, bar_color=clr.main[0],
                  xlim=(0.0, 1.1), xlabel='Feature Importance (loss: cross-entropy)',
                  ylim=(-1, len(top_means)), yticks=ticks, yticklabels=top_ticks_labels)

    plt.show()

    return PlotWrapper(fig, (ax,), {"features": ticks_labels, "stats": np.array(means)})


def _ale_num(df, model, feature_column, assembler, model_type, quantiles):
    """Compute ale from quantiles"""
    bins = spark.createDataFrame(list(zip(quantiles[:-1], quantiles[1:], list(range(len(quantiles) - 1)))),
                                 schema=StructType([StructField("lower", DoubleType()),
                                                    StructField("upper", DoubleType()),
                                                    StructField("interval", IntegerType())]))

    df = df.join(bins, ((df[feature_column] >= bins.lower) & (df[feature_column] < bins.upper)), how='left')
    df = df.fillna(quantiles[-2], subset='lower')\
           .fillna(quantiles[-1], subset='upper')\
           .fillna(len(quantiles) - 2, subset='interval')
    df = df.drop(feature_column)

    if model_type == 'classification':
        lower = df.withColumnRenamed('lower', feature_column)
        lower_pred = model.transform(assembler.transform(lower)).select(p1_proba('probability').alias('lower_pred'),
                                                                        F.col('_id_'), F.col('interval'))
        upper = df.withColumnRenamed('upper', feature_column)
        upper_pred = model.transform(assembler.transform(upper)).select(p1_proba('probability').alias('upper_pred'),
                                                                        F.col('_id_'))
    elif model_type == 'regression':
        lower = df.withColumnRenamed('lower', feature_column)
        lower_pred = model.transform(assembler.transform(lower)).select(F.col('prediction').alias('lower_pred'),
                                                                        F.col('_id_'), F.col('interval'))
        upper = df.withColumnRenamed('upper', feature_column)
        upper_pred = model.transform(assembler.transform(upper)).select(F.col('prediction').alias('upper_pred'),
                                                                        F.col('_id_'))
    else:
        raise ValueError("model_type can only be classification or regression")

    pred_diff = lower_pred.join(upper_pred, on='_id_')\
                          .withColumn('diff', F.col('upper_pred') - F.col('lower_pred'))
    ale_result = pred_diff.groupBy('interval').agg(F.avg('diff').alias('ale'), F.count('diff').alias('cnt'))
    interval, ale_temp, weights_temp = ale_result.toPandas().values.T

    ale = np.zeros((len(quantiles) - 1),)
    weights = np.zeros((len(quantiles) - 1),)
    ale[interval.astype(np.int64)] = ale_temp
    weights[interval.astype(np.int64)] = weights_temp

    ale = ale.cumsum()
    return ale - np.average(ale, weights=weights), weights


def feature_ale_plot(df, y_column, model, feature_column, model_type='classification',
                     model_input_col='features', columns_to_exclude=(), bins=100):
    """This function create the Accumulated Local Effect (ALE) plot of the target feature.
       Visit https://christophm.github.io/interpretable-ml-book/ale.html for more detailed explanation of ALE.

    Parameters
    ----------
    df : DataFrame
        Data to be plotted

    y_column : str
        Name of the class column

    model : pyspark.ml model
        The model object to be evaluated

    feature_column : str
        Name of the feature column to plot ALE on

    model_type : str, optional (default='classification')
        The type of model, 'classification' or 'regression'. For classification, the predicted positive class
        probability will be used; for regression, the predicted value will be used.

    model_input_col : str, optional (default='features')
        The name of the input column of the model, this is also the name of the output column of the VectorAssembler
        that creates the feature vector

    columns_to_exclude : tuple, optional (default=())
        Names of unwanted columns

    bins : int, optional (default=100)
        The number of intervals for the ALE plot

    Returns
    -------
    plot_wrapper : pytalite_spark.plotwrapper.PlotWrapper
        The PlotWrapper object that contains the information and data of the plot
    """
    # Get X, y array representation and feature indices from data
    df, assembler = preprocess(df, y_column, model_input_col, columns_to_exclude, create_id=True)

    feature_vals = df.select(feature_column).filter(F.col(feature_column) != -1)

    quantiles = np.array(feature_vals.distinct().approxQuantile(feature_column, list(np.arange(bins + 1) / bins), 1e-6))
    ale, counts = _ale_num(df, model, feature_column, assembler, model_type, quantiles.tolist())

    xs = (quantiles[1:] + quantiles[:-1]) / 2

    with plt.style.context(style_path):
        fig = plt.figure(figsize=(12, 9), facecolor=(1, 1, 1, 0))
        grid = GridSpec(2, 1, height_ratios=[10, 1], hspace=0)

        ax1 = plt.subplot(grid[0])
        fig.add_subplot(ax1)
        line_plot(ax1, xs, ale, line_label=False, xticks=[], ylabel="ALE of %s" % y_column)

        ax2 = plt.subplot(grid[1])
        fig.add_subplot(ax2, sharex=ax1)
        event_plot(ax2, feature_vals.toPandas().values.flatten(), 0.5, 1,
                   xlabel=feature_column, yticks=[], ylim=(-0.2, 1.2))
    plt.show()

    return PlotWrapper(fig, (ax1, ax2), {"quantiles": quantiles, "ale": ale, "quantile_distribution": counts})


def _partial_dependence_cat(df, model, feature_column, assembler, model_type, n_samples):
    """Compute PD values for categorical feature"""
    if type(feature_column) is str:
        pivot_col = feature_column
        feature_df = df.select(feature_column)
        df = df.drop(feature_column)
    else:
        pivot_col = feature_column[1]
        feature_df = df.select(feature_column[0], F.col(feature_column[1]).alias(pivot_col))
        df = df.drop(*feature_column)

    counts = feature_df.groupBy(pivot_col) \
        .agg(F.count(pivot_col).alias('cnt')) \
        .orderBy(pivot_col) \
        .select('cnt') \
        .toPandas().values.flatten()

    feature_df = feature_df.distinct()
    feature_df.cache()

    result_df = spark.createDataFrame(sc.emptyRDD(), schema=StructType([StructField(pivot_col, DoubleType()),
                                                                        StructField('pd', DoubleType())]))

    for i in range(n_samples):
        cross = df.sample(withReplacement=False, fraction=1 / n_samples).crossJoin(feature_df)
        if model_type == 'classification':
            pred = model.transform(assembler.transform(cross)).select(pivot_col, p1_proba('probability').alias('pred'))
        else:
            pred = model.transform(assembler.transform(cross)).select(pivot_col, F.col('prediction').alias('pred'))
        pd = pred.groupBy(pivot_col).agg(F.avg('pred').alias('pd'))
        result_df = result_df.union(pd)

    stats = result_df.groupBy(pivot_col).agg(F.avg('pd'), F.stddev('pd')).orderBy(pivot_col).toPandas().values
    vals = stats[:, 0]
    means = stats[:, 1].astype(np.float64)
    cis = np.apply_along_axis(lambda arr: st.norm.interval(0.95, loc=arr[0], scale=arr[1] / np.sqrt(n_samples)),
                              1, stats[:, 1:])

    return vals, counts, means, cis


def _partial_dependence_num(vals, df, model, feature_column, assembler, model_type, n_samples):
    """Compute PD values for numerical feature"""
    vals_df = spark.createDataFrame(list(map(lambda x: (float(x),), vals)),
                                    schema=StructType([StructField(feature_column, DoubleType())]))
    vals_df.cache()

    result_df = spark.createDataFrame(sc.emptyRDD(), schema=StructType([StructField(feature_column, DoubleType()),
                                                                        StructField('pd', DoubleType())]))

    df = df.drop(feature_column)

    for i in range(n_samples):
        cross = df.sample(withReplacement=False, fraction=1 / n_samples).crossJoin(vals_df)
        if model_type == 'classification':
            pred = model.transform(assembler.transform(cross))\
                        .select(feature_column, p1_proba('probability').alias('pred'))
        else:
            pred = model.transform(assembler.transform(cross))\
                        .select(feature_column, F.col('prediction').alias('pred'))
        pd = pred.groupBy(feature_column).agg(F.avg('pred').alias('pd'))
        result_df = result_df.union(pd)

    stats = result_df.groupBy(feature_column).agg(F.avg('pd'), F.stddev('pd')).orderBy(feature_column).toPandas().values
    means = stats[:, 1]
    cis = np.apply_along_axis(lambda arr: st.norm.interval(0.95, loc=arr[0], scale=arr[1] / np.sqrt(n_samples)),
                              1, stats[:, 1:])

    return means, cis


def _partial_dependence_plot_cat(df, y_column, model, feature_column, assembler, model_type, n_samples):
    """Create PDP for categorical feature"""
    vals, counts, means, cis = _partial_dependence_cat(df, model, feature_column, assembler, model_type, n_samples)
    xlabel = feature_column if type(feature_column) is str else feature_column[1]

    try:
        vals = vals.astype(np.float64)
        xticklabels = list(map(lambda x: "Cat. %d" % int(x), vals))
    except ValueError:
        xticklabels = vals

    indices = np.arange(len(vals))

    with plt.style.context(style_path):
        fig = plt.figure(figsize=(12, 9))
        grid = GridSpec(2, 1, height_ratios=[7, 0.5], hspace=0.1)
        ax1 = plt.subplot(grid[0])
        fig.add_subplot(ax1)

        line_plot(ax1, indices, cis[:, 0], line_color=clr.main[0], alpha=0.5, style='--', line_label=False)
        line_plot(ax1, indices, cis[:, 1], line_color=clr.main[0], alpha=0.5, style='--', line_label=False)
        line_plot(ax1, indices, means, marker='o', xticks=indices, xticklabels=[],
                  ylabel="Mean response of probability (%s)" % y_column,
                  xlim=(indices.min() - 0.5, indices.max() + 0.5),
                  ylim=(cis.min() - np.ptp(cis) * 0.2, cis.max() + np.ptp(cis) * 0.2))

        ax2 = plt.subplot(grid[1])
        fig.add_subplot(ax1, sharex=ax1)
        count_plot(ax2, counts, xticklabels=xticklabels, xlabel=xlabel, yticks=[])

    plt.show()

    return PlotWrapper(fig, (ax1, ax2), {"categories": xticklabels, "pd_avg": means,
                                         "pd_ci": cis, "cat_distribution": counts})


def _partial_dependence_plot_num(df, y_column, model, feature_column, assembler, model_type, n_samples):
    """Create PDP for numerical feature"""
    feature_vals = df.select(feature_column).filter(F.col(feature_column) != -1)
    feature_vals.cache()

    quantiles = np.array(feature_vals.distinct().approxQuantile(feature_column, list(np.arange(101) / 100), 1e-6))

    xs = (quantiles[1:] + quantiles[:-1]) / 2
    counts = []

    for i in range(len(quantiles) - 1):
        # Last interval needs to be inclusive at both boundaries
        if i != len(quantiles) - 2:
            count = df.filter((F.col(feature_column) >= quantiles[i]) &
                              (F.col(feature_column) < quantiles[i + 1])).count()
        else:
            count = df.filter((F.col(feature_column) >= quantiles[i]) &
                              (F.col(feature_column) <= quantiles[i + 1])).count()
        counts.append(count)

    counts = np.array(counts)
    means, cis = _partial_dependence_num(xs, df, model, feature_column, assembler, model_type, n_samples)

    with plt.style.context(style_path):
        fig = plt.figure(figsize=(12, 9), facecolor=(1, 1, 1, 0))
        grid = GridSpec(2, 1, height_ratios=[10, 1], hspace=0)

        ax1 = plt.subplot(grid[0])
        fig.add_subplot(ax1)
        line_plot(ax1, xs, cis[:, 0], line_color=clr.main[0], alpha=0.5, style='--', line_label=False)
        line_plot(ax1, xs, cis[:, 1], line_color=clr.main[0], alpha=0.5, style='--', line_label=False)
        line_plot(ax1, xs, means, line_label=False, xticks=[], ylabel="Mean response of %s" % y_column)

        ax2 = plt.subplot(grid[1])
        fig.add_subplot(ax2, sharex=ax1)
        event_plot(ax2, feature_vals.toPandas().values.flatten(), 0.5, 1,
                   xlabel=feature_column, yticks=[], ylim=(-0.2, 1.2))

    plt.show()

    return PlotWrapper(fig, (ax1, ax2), {"quantiles": quantiles, "pd_avg": means,
                                    "pd_ci": cis, "quantile_distribution": counts})


def partial_dependence_plot(df, y_column, model, feature_column, model_type='classification',
                            model_input_col='features', columns_to_exclude=(), n_samples=10):
    """This function create the Partial Dependence plot (PDP) of the target feature.
       Visit https://christophm.github.io/interpretable-ml-book/pdp.html for more detailed explanation of PDP.
       Note: this function uses bootstrap.

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

    model_type : str, optional (default='classification')
        The type of model, 'classification' or 'regression'. For classification, the predicted positive class
        probability will be used; for regression, the predicted value will be used.

    model_input_col : str, optional (default='features')
        The name of the input column of the model, this is also the name of the output column of the VectorAssembler
        that creates the feature vector

    columns_to_exclude : tuple, optional (default=())
        Names of unwanted columns

    n_samples : int, optional (default=10)
        The number of samples to bootstrap

    Returns
    -------
    plot_wrapper : pytalite_spark.plotwrapper.PlotWrapper
        The PlotWrapper object that contains the information and data of the plot
    """
    df, assembler = preprocess(df, y_column, model_input_col, columns_to_exclude)

    if type(feature_column) is str:
        if _feature_type(df, feature_column) == "categorical":
            return _partial_dependence_plot_cat(df, y_column, model, feature_column, assembler, model_type, n_samples)
        else:
            return _partial_dependence_plot_num(df, y_column, model, feature_column, assembler, model_type, n_samples)
    else:
        return _partial_dependence_plot_cat(df, y_column, model, feature_column, assembler, model_type, n_samples)
