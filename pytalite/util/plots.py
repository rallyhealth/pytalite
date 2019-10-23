from __future__ import absolute_import, division, print_function

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as lines
import matplotlib.ticker as ticker
import numpy as np

import pytalite.util.color as clr


__all__ = ['barh_plot', 'bar_plot', 'config_axes', 'count_plot', 'event_plot', 'line_plot', 'violin_plot']


def _safe_set_prop(setter, value, **kwargs):
    if value is not None:
        setter(value, **kwargs)


def config_axes(ax,
                xlabel=None, xlabelkw={}, xticks=None, xticklabels=None, xlim=None, xticklabelkw={}, x_invert=False,
                ylabel=None, ylabelkw={}, yticks=None, yticklabels=None, ylim=None, yticklabelkw={}, y_invert=False,
                show_legend=False, legendkw={}, auto_ticks=False):

    # X-axis
    _safe_set_prop(ax.set_xlim, xlim)

    if not ax.xaxis_inverted() and x_invert:
        ax.invert_xaxis()

    if xlim is not None and auto_ticks:
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        xticklabels_auto = list(map(lambda x: str(int(x)) if int(x) == x else str(round(x, 3)), ax.get_xticks()))
    else:
        xticklabels_auto = None

    if xticks is not None and xticklabels is None:
        xticklabels = list(map(lambda x: str(int(x)) if int(x) == x else str(round(x, 3)), xticks))
    elif xticks is None and xticklabels is None:
        xticklabels = xticklabels_auto

    if len(xticklabelkw) != 0 and xticklabels is None:
        xticklabels = list(map(lambda x: x.get_text(), ax.get_xticklabels()))

    if len(xlabelkw) != 0 is not None and xlabel is None:
        xlabel = ax.get_xlabel()

    _safe_set_prop(ax.set_xlabel, xlabel, **xlabelkw)
    _safe_set_prop(ax.set_xticks, xticks)
    _safe_set_prop(ax.set_xticklabels, xticklabels, **xticklabelkw)

    # Y-axis
    _safe_set_prop(ax.set_ylim, ylim)

    if not ax.yaxis_inverted() and y_invert:
        ax.invert_yaxis()

    if ylim is not None and auto_ticks:
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        yticklabels_auto = list(map(lambda y: str(int(y)) if int(y) == y else str(round(y, 3)), ax.get_yticks()))
    else:
        yticklabels_auto = None

    if yticks is not None and yticklabels is None:
        yticklabels = list(map(lambda y: str(int(y)) if int(y) == y else str(round(y, 3)), yticks))
    elif yticks is None and yticklabels is None:
        yticklabels = yticklabels_auto

    if len(yticklabelkw) != 0 and yticklabels is None:
        yticklabels = list(map(lambda y: y.get_text(), ax.get_yticklabels()))

    if len(ylabelkw) != 0 and ylabel is None:
        ylabel = ax.get_ylabel()

    _safe_set_prop(ax.set_ylabel, ylabel, **ylabelkw)
    _safe_set_prop(ax.set_yticks, yticks)
    _safe_set_prop(ax.set_yticklabels, yticklabels, **yticklabelkw)

    if show_legend or len(legendkw) != 0:
        prop = legendkw.get('fontproperties')
        color = legendkw.get('color')
        legend = ax.legend(prop=prop)
        for text in legend.get_texts():
            _safe_set_prop(text.set_color, color)


def _bar_auto_label(ax, bars, errs=None):
    if errs is None:
        errs = [None] * len(bars)

    for bar, err in zip(bars, errs):
        height = bar.get_height()
        bbox_props = {'facecolor': (1, 1, 1, .5), 'edgecolor': ax.xaxis.label.get_color(), 'boxstyle': "square,pad=0.5"}

        text_y = height + err + ax.get_ylim()[1] / 15 if err is not None else height
        ax.text(bar.get_x() + bar.get_width() / 2., text_y, '%d' % int(height),
                ha='center', va='center', bbox=bbox_props, color=ax.xaxis.label.get_color())


def bar_plot(ax, x, height, width=-1, align='center', bar_color=clr.main[1], edge_color=None, bar_label=True, **kwargs):
    if width == -1:
        width = min(0.4, 0.4 / (10.0 / len(height)))

    bars = ax.bar(x, height, width, align=align, color=bar_color, edgecolor=edge_color, alpha=0.75)
    config_axes(ax, **kwargs)

    if bar_label:
        _bar_auto_label(ax, bars)


def _barh_auto_label(ax, bars, errs=None):
    if errs is None:
        upper_errors = [None] * len(bars)
    else:
        upper_errors = errs[1]

    for bar, err in zip(bars, upper_errors):
        width = bar.get_width()
        bbox_props = {'facecolor': (1, 1, 1, .5), 'edgecolor': ax.xaxis.label.get_color(), 'boxstyle': "square,pad=0.5"}

        text_x = width + ax.get_xlim()[1] / 13 + (err if err is not None else 0)
        ax.text(text_x, bar.get_y() + bar.get_height() / 2., '%.3f' % width,
                ha='right', va='center', bbox=bbox_props, color=ax.xaxis.label.get_color())


def barh_plot(ax, x, width, height=-1, align='center', bar_color=clr.main[1], edge_color=None, bar_label=True,
              xerr=None, **kwargs):
    if height == -1:
        height = min(0.7, 0.7 / (10.0 / len(width)))

    bars = ax.barh(x, width, height, align=align, color=bar_color, edgecolor=edge_color, xerr=xerr, alpha=0.75)
    config_axes(ax, **kwargs)

    if bar_label:
        _barh_auto_label(ax, bars, xerr)


def count_plot(ax, counts, color_map=cm.get_cmap("Blues"), **kwargs):
    norm = mcolors.Normalize(vmin=0, vmax=counts.max())
    kwargs['xticks'] = np.arange(counts.shape[0])

    ax.imshow(np.expand_dims(counts, 0), aspect="auto", cmap=color_map, norm=norm, alpha=0.75,
              extent=(-0.5, kwargs['xticks'].max() + 0.5, 0, 0.5))
    config_axes(ax, **kwargs)

    for x in kwargs['xticks'][:-1]:
        ax.add_line(lines.Line2D([x + 0.5, x + 0.5], [0, 1], color="w", linestyle='-', linewidth=1.5))

    fractions = counts * 100 / counts.sum()

    for i in range(counts.shape[0]):
        text_color = "black"
        if counts[i] >= counts.max() / 2:
            text_color = "white"
        ax.text(kwargs['xticks'][i], 0.25, "%.2f%%" % fractions[i], ha="center", va="center", color=text_color)


def _line_auto_label(ax, x_data, y_data):
    bbox_props = {'facecolor': (1, 1, 1, 0), 'edgecolor': ax.xaxis.label.get_color(), 'boxstyle': "square,pad=0.5"}

    ylim = ax.get_ylim()

    for x, y in zip(x_data, y_data):
        ax.text(x, y + (ylim[1] - ylim[0]) / 15, '%.3f' % y, ha="center", va="top", size=10,
                bbox=bbox_props, color=ax.xaxis.label.get_color())


def line_plot(ax, x_data, y_data, line_color=clr.main[1], alpha=1.0,  marker=None, style='-', width=2, line_label=True,
              legend=None, **kwargs):
    if legend is not None:
        kwargs.setdefault('show_legend', True)

    ax.plot(x_data, y_data, color=line_color, alpha=alpha, marker=marker, ls=style, lw=width, label=legend)
    config_axes(ax, **kwargs)

    if line_label:
        _line_auto_label(ax, x_data, y_data)


def event_plot(ax, dist_data, lineoffsets, linelengths, color=clr.main[1], **kwargs):
    ax.eventplot(dist_data, lineoffsets=lineoffsets, linelengths=linelengths, linewidths=0.4, color=color, alpha=0.75)
    config_axes(ax, **kwargs)


def violin_plot(ax, data, positions, violin_color=clr.main[1], bar_color=clr.main[1], **kwargs):
    parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=False, showextrema=False)

    for pc in parts['bodies']:
        pc.set_facecolor(violin_color)
        pc.set_edgecolor(bar_color)
        pc.set_alpha(0.5)

    quartile1, medians, quartile3 = zip(*map(lambda d: np.percentile(d, [25, 50, 75]), data))

    ax.scatter(positions, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(positions, quartile1, quartile3, color=bar_color, linestyle='-', lw=5)

    config_axes(ax, **kwargs)
