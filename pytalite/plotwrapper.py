from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt

from pytalite.util.plots import config_axes


class PlotWrapper(object):
    """Class to wrap plot data and attributes"""
    _kw_postfix_dict = {'%s_fp': 'fontproperties', '%s_color': 'color', '%s_rotation': 'rotation'}

    def __init__(self, fig, axes, data):
        self.fig = fig
        self.axes = axes
        self.data = data

    def adjust_plot(self, *axes_args, **kwargs):
        """Adjust the plot

        Parameters
        ----------
        axes_args : dicts
            Dictionaries that contains the new property of the axes. If the dicts is not specifically labeled for their
            axes (i.e. does not have 'ax_id' entry), by default, the first dict corresponds to the first axes.

        **kwargs
            The only valid option here is 'fig_size', which re-sizes the plot.
        """
        if 'fig_size' in kwargs:
            new_width, new_height = kwargs.get('fig_size')
            curr_width, _ = self.fig.get_size_inches()
            self.fig.set_size_inches(new_width, new_height)
            self.fig.set_dpi(self.fig.get_dpi() * curr_width / new_width)

        for ax_id, ax_args in enumerate(axes_args):
            if 'ax_id' in ax_args:
                ax = self.axes[ax_args['ax_id']]
                del ax_args['ax_id']
            else:
                ax = self.axes[ax_id]

            xlabelkw = self._pack_kwargs('xlabel', ax_args)
            ylabelkw = self._pack_kwargs('ylabel', ax_args)
            xticklabelkw = self._pack_kwargs('xticklabel', ax_args)
            yticklabelkw = self._pack_kwargs('yticklabel', ax_args)
            legendkw = self._pack_kwargs('legend', ax_args)

            config_axes(ax, xlabelkw=xlabelkw, ylabelkw=ylabelkw, xticklabelkw=xticklabelkw, yticklabelkw=yticklabelkw,
                        auto_ticks=True, legendkw=legendkw, **ax_args)

    @staticmethod
    def _pack_kwargs(prefix, ax_args):
        kwargs = {}
        for postfix, internal_kw_name in PlotWrapper._kw_postfix_dict.items():
            user_kw_name = postfix % prefix
            if user_kw_name in ax_args:
                kwargs[internal_kw_name] = ax_args[user_kw_name]
                del ax_args[user_kw_name]
        return kwargs

    def display_plot(self):
        """Display the plot"""
        color_str, (width, height) = self.fig.canvas.print_to_buffer()
        color_arr = np.fromstring(color_str, np.uint8).reshape((height, width, 4))
        sizes = np.shape(color_arr)
        fig = plt.figure()
        fig.set_size_inches(9 * sizes[1] / sizes[0], 9, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(color_arr)

    def save_plot(self, path, **kwargs):
        """Save the plot to path

        Parameters
        ----------
        path : str
            The save path

        **kwargs
        keyword arguments are passed to Figure.savefig call
        """
        kwargs.setdefault('facecolor', self.fig.get_facecolor())
        kwargs.setdefault('edgecolor', self.fig.get_edgecolor())
        kwargs.setdefault('bbox_inches', "tight")

        self.fig.savefig(path, **kwargs)
