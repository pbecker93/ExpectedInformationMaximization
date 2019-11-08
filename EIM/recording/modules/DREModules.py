from recording.Recorder import RecorderModule, plt
import numpy as np
import tensorflow as tf
import os
"""Modules to record Density Ratio Estimator training"""

class DRERecMod(RecorderModule):
    """Records current Density Ratio Estimator performance - loss, accuracy and mean output for true and fake samples"""
    def __init__(self, true_samples, target_ld=None):
        super().__init__()
        if isinstance(true_samples, np.ndarray):
            self._target_samples = true_samples.astype(np.float32)
        else:
            self._target_samples = [x.astype(np.float32) for x in true_samples]
        self._steps = []
        self._estm_ikl = []
        self._loss = []
        self._acc = []
        self._true_mean = []
        self._fake_mean = []
        self.__num_iters = None
        self._target_ld = target_ld
        self._dre_type = None
        if self._target_ld is not None:
            self._dre_rmse = []
        self._conditional = not isinstance(true_samples, np.ndarray)

    def initialize(self, recorder, plot_realtime, save, num_iters):
        super().initialize(recorder, plot_realtime, save)
        self.__num_iters = num_iters

    @property
    def _num_iters(self):
        assert self.__num_iters is not None, "Density Ratio Estimator Recorder not properly initialized"
        return self.__num_iters

    def _dre_rmse_fn(self, idx, dre, model, samples):
        mld = model.log_density(samples)
        return tf.squeeze(dre(samples, idx)) - (self._target_ld(samples) - mld)

    def record(self, dre, model, iteration, steps):
        if iteration == 0 and self._target_ld is not None:
            for _ in model.components:
                self._dre_rmse.append([])
        estm_ikl, loss, acc, true_mean, fake_mean = [np.array(x) for x in dre.eval(self._target_samples, model)]
        log_str = "Density Ratio Estimator ran for " + str(steps) + " steps. "
        log_str += "Loss {:.4f} ".format(loss)
        log_str += "Estimated IKL: {:.4f} ".format(estm_ikl)
        log_str += "Accuracy: {:.4f} ".format(acc)
        log_str += "True Mean IKL: {:.4f} ".format(true_mean)
        log_str += "Fake Mean IKL: {:.4f} ".format(fake_mean)

        if self._target_ld is not None:
            all_errs = []
            for i, c in enumerate(model.components):
                samples = c.sample(1000)
                errs = np.array(self._dre_rmse_fn(i, dre, model, samples))
                all_errs.append(errs)
                self._dre_rmse[i].append(np.sqrt(np.mean(errs**2)))
                log_str += "Component {:d}: DRE RMSE: {:.4f} ".format(i, self._dre_rmse[i][-1])
            self._recorder.handle_plot("Err Hist", self._plot_hist, all_errs)
        self._logger.info(log_str)

        self._steps.append(steps)
        self._estm_ikl.append(estm_ikl)
        self._loss.append(loss)
        self._acc.append(acc)
        self._true_mean.append(true_mean)
        self._fake_mean.append(fake_mean)
        if self._plot_realtime:
            self._recorder.handle_plot("Discriminator Evaluation", self._plot)

    def finalize(self):
        if self._save:
            save_dict = {"estm_ikl": self._estm_ikl, "loss": self._loss, "acc": self._acc, "true_mean":
                         self._true_mean, "fake_mean": self._fake_mean}
            np.savez_compressed(os.path.join(self._save_path, "DensityRatioEstimatorEval_raw.npz"), **save_dict)
            self._recorder.save_img("DensityRatioEstimatorEval", self._plot)

    def _subplot(self, i, title, data_list, data_list2=None, y_lim=None):
        plt.subplot(5 if self._target_ld is not None else 4, 1, i)
        plt.title(title)
        plt.plot(np.array(data_list))
        if data_list2 is not None:
            plt.plot(np.array(data_list2))
        plt.xlim(0, self._num_iters)
        if y_lim is not None:
            plt.ylim(y_lim[0], y_lim[1])

    def _plot_hist(self, errs):
        for i, err in enumerate(errs):
            plt.subplot(len(errs), 1, i+1)
            plt.hist(err, density=True, bins=25)
        plt.tight_layout()

    def _plot(self):
        self._subplot(1, "Estimated I-Projection", self._estm_ikl)
        self._subplot(2, "Density Ratio Estimator Loss", self._loss)
        self._subplot(3, "Density Ratio Estimator Accuracy", self._acc, y_lim=(-0.1, 1.1))
        self._subplot(4, "", self._true_mean, self._fake_mean, y_lim=(-0.1, 1.1))
        plt.legend(["Mean output true samples", "Mean output fake samples"])
        if self._target_ld is not None:
            plt.subplot(5, 1, 5)
            for i in range(len(self._dre_rmse)):
                self._subplot(5, "DRE RMSE", self._dre_rmse[i])
        plt.tight_layout()

    def get_last_rec(self):
        lr = {"steps": self._steps[-1], "estimated_ikl": self._estm_ikl[-1], "dre_loss": self._loss[-1],
              "accuracy": self._acc[-1], "true_mean": self._true_mean[-1], "fake_mean": self._fake_mean[-1]}
        if self._target_ld is not None:
            for i in range(len(self._dre_rmse)):
                lr["dre_rmse{:d}".format(i)] = self._dre_rmse[i][-1]
        return lr

    @property
    def logger_name(self):
        return "DRE"
