from recording.Recorder import RecorderModule, plt, Colors
import numpy as np

def log_res(res, key_prefix):
    num_iters, kl, entropy, add_text = [np.array(x) for x in res]
    last_rec = {key_prefix + "_num_iterations": num_iters, key_prefix + "_kl": kl, key_prefix + "_entropy": entropy}
    log_string = "Updated for {:d} iterations. ".format(num_iters)
    log_string += "KL: {:.5f}. ".format(kl)
    log_string += "Entropy: {:.5f} ".format(entropy)
    log_string += str(add_text)
    return log_string, last_rec


class WeightUpdateRecMod(RecorderModule):

    def __init__(self, plot):
        super().__init__()
        self._last_rec = None
        self._plot = plot
        self._kls = []
        self._entropies = []
        self._num_iters = -1

    def initialize(self, recorder, plot_realtime, save, num_iters):
        super().initialize(recorder, plot_realtime, save)
        self._num_iters = num_iters

    def record(self, res):
        log_string, self._last_rec = log_res(res, "weights")
        self._logger.info(log_string)
        if self._plot:
            self._kls.append(self._last_rec["weights_kl"])
            self._entropies.append(self._last_rec["weights_entropy"])
            self._recorder.handle_plot("Weight Update", self._plot_fn)

    def _plot_fn(self):
        plt.subplot(2, 1, 1)
        plt.title("Expected KL")
        plt.plot(self._kls)
        plt.xlim(0, self._num_iters)
        plt.subplot(2, 1, 2)
        plt.title("Expected Entropy")
        plt.plot(self._entropies)
        plt.xlim(0, self._num_iters)
        plt.tight_layout()

    @property
    def logger_name(self):
        return "Weight Update"

    def get_last_rec(self):
        assert self._last_rec is not None
        return self._last_rec

    def finalize(self):
        if self._plot:
            self._recorder.save_img("WeightUpdates", self._plot_fn)


class ComponentUpdateRecMod(RecorderModule):

    def __init__(self, plot, summarize=True):
        super().__init__()
        self._plot = plot
        self._last_rec = None
        self._summarize = summarize
        self._kls = None
        self._entropies = None
        self._num_iters = -1
        self._num_components = -1
        self._c = Colors()

    def initialize(self, recorder, plot_realtime, save, num_iters, num_components):
        super().initialize(recorder, plot_realtime, save)
        self._num_iters = num_iters
        self._num_components = num_components
        self._kls = [[] for _ in range(self._num_components)]
        self._entropies = [[] for _ in range(self._num_components)]

    def record(self, res_list):
        self._last_rec = {}
        for i, res in enumerate(res_list):
            cur_log_string, cur_last_rec = log_res(res, "component_{:d}".format(i))
            self._last_rec = {**self._last_rec, **cur_last_rec}
            if not self._summarize:
                self._logger.info("Component{:d}: ".format(i + 1) + cur_log_string)
            if self._plot:
                self._kls[i].append(self._last_rec["component_{:d}_kl".format(i)])
                self._entropies[i].append(self._last_rec["component_{:d}_entropy".format(i)])
        if self._summarize:
            self._summarize_results(res_list)
        if self._plot:
            self._recorder.handle_plot("Component Update", self._plot_fn)

    def _summarize_results(self, res_list):
        fail_ct = 0
        for res in res_list:
            if "failed" in str(res[-1]).lower():
                fail_ct += 1
        num_updt = len(res_list)
        log_str = "{:d} components updated - {:d} successful".format(num_updt, num_updt - fail_ct)
        self._logger.info(log_str)




    def _plot_fn(self):
        plt.subplot(2, 1, 1)
        plt.title("Expected KL")
        for i in range(self._num_components):
            plt.plot(self._kls[i], c=self._c(i))
        plt.legend(["Component {:d}".format(i + 1) for i in range(self._num_components)])
        plt.xlim(0, self._num_iters)
        plt.subplot(2, 1, 2)
        plt.title("Expected Entropy")
        for i in range(self._num_components):
            plt.plot(self._entropies[i], c=self._c(i))
        plt.xlim(0, self._num_iters)
        plt.tight_layout()

    @property
    def logger_name(self):
        return "Component Update"

    def get_last_rec(self):
        assert self._last_rec is not None
        return self._last_rec

    def finalize(self):
        if self._plot:
            self._recorder.save_img("ComponentUpdates", self._plot_fn)
