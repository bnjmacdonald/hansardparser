import os
import time
import sys
import numpy as np
import tensorflow as tf
# import utils
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

class PlotConfig(object):
    # plt.switch_backend('Agg')
    sns.set_style('whitegrid', {'axes.edgecolor': '.8', 'grid.color': '.95'})
    sns.set_context('paper', font_scale=1.5)
    dpi = 150
    subplot_titlesize = 20
    height = 3
    width = 6
    # color palette for pvalues.
    tableau10_cb = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89), (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236), (255, 188, 121), (207, 207, 207)]
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120), (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150), (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148), (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199), (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    tableau10_cb_hex = ['#%02x%02x%02x' % col for col in tableau10_cb]
    tableau20_hex = ['#%02x%02x%02x' % col for col in tableau20]
    base_blue = tableau10_cb_hex[0]
    # pvalue_palette2 = [base_blue, utils.get_pvalue_palette(3)[0]]
    sns.set_palette(sns.color_palette(tableau10_cb_hex))

plot_config = PlotConfig()


def orthogonal_initializer(scale=1.1):
    """From Lasagne and Keras.

    Reference: Saxe et al., http://arxiv.org/abs/1312.6120.
    """
    # print('Warning -- You have opted to use the orthogonal_initializer function')
    def _initializer(shape, dtype=tf.float32):
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape) #this needs to be corrected to float32
        print('you have initialized one orthogonal matrix.')
        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer


def heatmap_confusion(mat, outpath):
    """Plots a confusion matrix."""
    cmap = sns.cubehelix_palette(6, start=1, rot=0, light=.9, dark=0, reverse=False, as_cmap=True)
    sns.heatmap(mat, annot=True, linewidths=0.5, square=True, yticklabels=True, cmap=cmap)
    ax = plt.gca()
    plt.title('Confusion matrix', fontsize=plot_config.subplot_titlesize, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=plot_config.subplot_titlesize-10)
    plt.ylabel('True label', fontsize=plot_config.subplot_titlesize-10)
    plt.savefig(outpath, dpi=plot_config.dpi, bbox_inches='tight')
    plt.close()
    return ax


class Progbar(object):
    """Displays a progress bar.
    
    Progbar class copied from keras (https://github.com/fchollet/keras/).
    
    Arguments:
        
        target: Total number of steps expected.
        
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbosity=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbosity = verbosity

    def update(self, current, values=[], exact=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbosity > 0:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbosity > 1:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far+n, values)
