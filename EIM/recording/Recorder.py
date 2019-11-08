import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import os
import warnings
import logging
import sys

class _CWFormatter(logging.Formatter):
    def __init__(self):
        #self.std_formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        self.std_formatter = logging.Formatter('[%(name)s] %(message)s')
        self.red_formatter = logging.Formatter('[%(asctime)s] %(message)s')

    def format(self, record: logging.LogRecord):
        if record.levelno <= logging.ERROR:
            return self.std_formatter.format(record)
        else:
            return self.red_formatter.format(record)

sh = logging.StreamHandler(sys.stdout)
formatter = _CWFormatter()
sh.setFormatter(formatter)
sh.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, handlers=[sh])

class RecorderKeys:
    TRAIN_ITER = "train_iteration_module"
    INITIAL = "initial_module"
    MODEL = "model_module"
    DRE = "dre_rec_mod"
    WEIGHTS_UPDATE = "weights_update"
    COMPONENT_UPDATE = "component_update"

class Recorder:

    def __init__(self, modules_dict,
                 plot_realtime, save,
                 save_path=os.path.abspath("rec"),
                 fps=3, img_dpi=1200, vid_dpi=500):
        """
        Initializes and handles recording modules
        :param modules_dict: Dictionary containing all recording modules
        :param plot_realtime: whether to plot in realtime while algorithm is running
        :param save: whether to save plots/images
        :param save_path: path to save data to
        :param fps: video frame rate
        :param img_dpi: image resolution
        :param vid_dpi: video resolution
        """
        self.save_path = save_path
        self._modules_dict = modules_dict

        self._plot_realtime = plot_realtime
        self._save = save

        self._fps = fps
        self._img_dpi = img_dpi
        self._vid_dpi = vid_dpi

        if not os.path.exists(save_path):
             warnings.warn('Path ' + save_path + ' not found - creating')
             os.makedirs(save_path)

    def handle_plot(self, name, plot_fn, data=None):
        if self._plot_realtime:
            plt.figure(name)
            plt.clf()
            plot_fn() if data is None else plot_fn(data)
            plt.pause(0.0001)

    def save_img(self, name, plot_fn, data=None):
        """
        Saves an image
        :param name: file name
        :param plot_fn: function generating the plot
        :param data: data provided to plot_fn to generate plot
        :return:
        """
        fig = plt.figure(name)
        plot_fn() if data is None else plot_fn(data)
        fig.savefig(os.path.join(self.save_path, name + ".pdf"), format="pdf", dpi=self._img_dpi)
        plt.close(name)

    def save_vid(self, name, update_fn, frames):
        """
        Saves a video
        :param name: file name
        :param update_fn: function generating plots from data
        :param frames: list containing data for frames
        :return:
        """
        def _update_fn_wrapper(i):
            plt.clf()
            update_fn(i)

        if self._save:
            fig = plt.figure()
            ani = anim.FuncAnimation(fig,
                                     func=_update_fn_wrapper,
                                     frames=frames)
            writer = anim.writers['imagemagick'](fps=self._fps)
            ani.save(os.path.join(self.save_path, name+".mp4"),
                     writer=writer,
                     dpi=(500 if self._vid_dpi > 500 else self._vid_dpi))

    def save_vid_raw(self, name, data, preprocess_fn=None):
        save_dict = {}
        preprocess_fn = (lambda x: x) if preprocess_fn is None else preprocess_fn
        for i, d in enumerate(data):
            save_dict[str(i)] = preprocess_fn(data[i])
        np.savez(os.path.join(self.save_path, name + "_raw.npz"), **save_dict)

    def initialize_module(self, name, *args, **kwargs):
        module = self._modules_dict.get(name)
        if module is not None:
            module.initialize(self, self._plot_realtime, self._save, *args, **kwargs)

    def __call__(self, module, *args, **kwargs):
        module = self._modules_dict.get(module)
        if module is not None:
            module.record(*args, **kwargs)

    def snapshot(self):
        for key in self._modules_dict.keys():
            self._modules_dict[key].snapshot()

    def finalize_training(self):
        for key in self._modules_dict.keys():
            m = self._modules_dict[key]
            if m.is_initialized:
                m.finalize()

    def get_last_rec(self):
        last_rec = {}
        for key in self._modules_dict.keys():
            last_rec = {**last_rec, **self._modules_dict[key].get_last_rec()}
        return last_rec


class RecorderModule:
    """RecorderModule Superclass"""

    def __init__(self):
        self.__recorder = None
        self._plot_realtime = None
        self._save = None
        self._logger = logging.getLogger(self.logger_name)

    @property
    def _save_path(self):
        return self._recorder.save_path

    @property
    def _recorder(self):
        assert self.__recorder is not None, "recorder not set yet - Recorder module called before proper initialization "
        return self.__recorder

    @property
    def is_initialized(self):
        return self.__recorder is not None

    def initialize(self, recorder, plot_realtime, save, *args):
        self.__recorder = recorder
        self._save = save
        self._plot_realtime = plot_realtime

    def record(self, *args):
        raise NotImplementedError

    @property
    def logger_name(self):
        raise NotImplementedError

    def finalize(self):
        pass

    def get_last_rec(self):
        return {}


class Colors:
    """Provides colors for plotting """
    def __init__(self, pyplot_color_cycle=True):
        if pyplot_color_cycle:
            self._colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        else:
            #Todo implement color list with more colors...
            raise NotImplementedError("Not yet implemented")

    def __call__(self, i):
        return self._colors[i % len(self._colors)]



