from recording.Recorder import RecorderModule, plt, Colors
import numpy as np
from distributions import SaveAndLoad
import os

class ModelRecMod(RecorderModule):
    """Records current eim performance: Log Likelihood and if true (unnormalized) log density is provided the
    i-projection kl (plus constant)"""
    def __init__(self, train_samples, test_samples, true_log_density=None, eval_fn=None,
                 test_log_iters=50, save_log_iters=50):
        super().__init__()
        self._train_samples = train_samples
        self._test_samples = test_samples
        self._true_log_density = true_log_density
        self._eval_fn = eval_fn

        self._test_log_iters = test_log_iters
        self._save_log_iters = save_log_iters

        self._train_ll_list = []
        self._train_kl_list = []

        self._test_ll_list = []
        self._test_kl_list = []
        self._test_eval_list = []

        self.__num_iters = None
        self._colors = Colors()

    @property
    def _num_iters(self):
        assert self.__num_iters is not None, "Model Recorder not initialized properly"
        return self.__num_iters

    def initialize(self, recorder, plot_realtime, save, num_iters):
        super().initialize(recorder, plot_realtime, save)
        self.__num_iters = num_iters

    def _log_train(self, model):
        if isinstance(self._train_samples, np.ndarray):
            self._train_ll_list.append(np.array(model.log_likelihood(self._train_samples)))
        else:
            self._train_ll_list.append(np.array(model.log_likelihood(self._train_samples[0], self._train_samples[1])))
        log_str = "Training LL: " + str(self._train_ll_list[-1])
        if self._true_log_density is not None:
            if isinstance(self._train_samples, np.ndarray):
                model_train_samples = model.sample(len(self._train_samples))
                self._train_kl_list.append(np.mean(model.log_density(model_train_samples) -
                                                   self._true_log_density(model_train_samples)))
            else:
                model_train_samples = model.sample(self._train_samples[0])
                self._train_kl_list.append(np.mean(model.log_density(self._train_samples[0], model_train_samples) -
                                                   self._true_log_density(self._train_samples[0], model_train_samples)))

            log_str += " KL (MC-Estimate): " + str(self._train_kl_list[-1])
        self._logger.info(log_str)
        if self._plot_realtime:
            self._recorder.handle_plot("Loss", self._plot)

    def _log_test(self, model, model_test_samples=None):
        if isinstance(self._test_samples, np.ndarray):
            self._test_ll_list.append(np.array(model.log_likelihood(self._test_samples)))
        else:
            self._test_ll_list.append(np.array(model.log_likelihood(self._test_samples[0], self._test_samples[1])))

        log_str = "Test: Likelihood " + str(self._test_ll_list[-1])

        if self._true_log_density is not None:
            if isinstance(self._train_samples, np.ndarray):
                model_test_samples = model.sample(len(self._test_samples))
                self._test_kl_list.append(np.mean(model.log_density(model_test_samples) -
                                                   self._true_log_density(model_test_samples)))
            else:
                model_test_samples = model.sample(self._test_samples[0])
                self._test_kl_list.append(np.mean(model.log_density(self._test_samples[0], model_test_samples) -
                                                   self._true_log_density(self._test_samples[0], model_test_samples)))
            log_str += " KL (MC-Estimate): " + str(self._test_kl_list[-1])
        if self._eval_fn is not None:
            self._test_eval_list.append(self._eval_fn(model))
            log_str += " Eval Loss: " + str(self._test_eval_list[-1])
        self._logger.info(log_str)

    def record(self, model, iteration):
        if self._save and (iteration % self._save_log_iters == 0):
            SaveAndLoad.save_model(model, self._save_path, "modelAtIter{:05d}".format(iteration))
        self._log_train(model)
        if iteration % self._test_log_iters == 0:
            self._log_test(model)

    def _plot(self):
        plt.subplot(2 if self._true_log_density is not None else 1, 1, 1)
        plt.title("Train Log Likelihood")

        plt.plot(np.arange(0, len(self._train_ll_list)), np.array(self._train_ll_list))
        plt.xlim(0, self._num_iters)
        if self._true_log_density is not None:
            plt.subplot(2, 1, 2)
            plt.title("I-Projection KL (MC-Estimate)")
            plt.plot(np.arange(0, len(self._train_kl_list)), np.array(self._train_kl_list))
            plt.xlim((0, self._num_iters))
        plt.tight_layout()

    def finalize(self):
        if self._save:
            save_dict = {"ll_train": self._train_ll_list,
                         "ll_test": self._test_ll_list}
            if self._true_log_density is not None:
                save_dict["kl_train"] = self._train_kl_list
                save_dict["kl_test"] = self._test_kl_list
            if self._eval_fn is not None:
                save_dict["eval_test"] = self._test_eval_list
            np.savez_compressed(os.path.join(self._save_path, "losses_raw.npz"), **save_dict)
            self._recorder.save_img("Loss", self._plot)

    def get_last_rec(self):
        res_dict = {"ll_train": self._train_ll_list[-1]}
        if self._true_log_density is not None:
            res_dict["kl_train"] = self._train_kl_list[-1]

        return res_dict

    @property
    def logger_name(self):
        return "Model"


class ModelRecModWithModelVis(ModelRecMod):
    """Superclass - Standard eim recording + eim visualization
    (actual visualization needs to be implemented individually depending on data/task)"""

    def record(self, model, iteration):
        super().record(model, iteration)
        if self._plot_realtime:
            plt_fn = lambda x: self._plot_model(x, title="Iteration {:5d}".format(iteration))
            self._recorder.handle_plot("Model", plt_fn, model)

    def _plot_model(self, model, title):
        raise NotImplementedError("Not Implemented")

    @staticmethod
    def _draw_2d_covariance(mean, covmatrix, chisquare_val=2.4477, return_raw=False, *args, **kwargs):
        (largest_eigval, smallest_eigval), eigvec = np.linalg.eig(covmatrix)
        phi = -np.arctan2(eigvec[0, 1], eigvec[0, 0])

        a = chisquare_val * np.sqrt(largest_eigval)
        b = chisquare_val * np.sqrt(smallest_eigval)

        ellipse_x_r = a * np.cos(np.linspace(0, 2 * np.pi))
        ellipse_y_r = b * np.sin(np.linspace(0, 2 * np.pi))

        R = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
        r_ellipse = np.array([ellipse_x_r, ellipse_y_r]).T @ R
        if return_raw:
            return mean[0] + r_ellipse[:, 0], mean[1] + r_ellipse[:, 1]
        else:
            return plt.plot(mean[0] + r_ellipse[:, 0], mean[1] + r_ellipse[:, 1], *args, **kwargs)


class GMM2DModelRecMod(ModelRecModWithModelVis):
    """Visualization for 2 Dimensional Mixture of Gaussians"""
    def __init__(self, train_samples, test_samples, true_means=None, true_covars=None, true_log_density=None,
                 eval_fn=None, test_log_iters=50, save_log_iters=50):
        super().__init__(train_samples, test_samples, true_log_density, eval_fn, test_log_iters, save_log_iters)
        self._models = []
        self._true_means = true_means
        self._true_covars = true_covars

    def _plot_model(self, model, title):
        plt.title(title)
        line_list = []
        label_list = []
        for i, c in enumerate(model.components):
            l, = self._draw_2d_covariance(c.mean, c.covar, c=self._colors(i + 1))
            line_list.append(l)
            label_list.append(np.array(model.weight_distribution.probabilities)[i])
        plt.legend(line_list, label_list)

        if self._true_means is not None:
            for i in range(len(self._true_means)):
                self._draw_2d_covariance(self._true_means[i], self._true_covars[i], c=self._colors(0))


class ObstacleModelRecMod(ModelRecModWithModelVis):

    def __init__(self, obstacle_data, train_samples, test_samples, true_log_density=None, eval_fn=None,
                 test_log_iters=50, save_log_iters=50):
        super().__init__(train_samples, test_samples, true_log_density, eval_fn, test_log_iters, save_log_iters)
        self._data = obstacle_data

    def _plot_model(self, model, title):
        x_plt = np.arange(0, 1, 1e-2)
        color = Colors()
        contexts = self._data.raw_test_samples[0][:10]
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            context = contexts[i:i + 1]
            plt.imshow(self._data.img_from_context(context[0]))
            lines = []
            for k, c in enumerate(model.components):
                m = (c.mean(context)[0] + 1) / 2
                cov = c.covar(context)[0]
                mx, my = m[::2], m[1::2]
                plt.scatter(200 * mx, 100 * my, c=color(k))
                for j in range(mx.shape[0]):
                    mean = np.array([mx[j], my[j]])
                    cov_j = cov[2 * j: 2 * (j + 1), 2 * j: 2 * (j + 1)]
                    plt_cx, plt_cy = self._draw_2d_covariance(mean, cov_j, 1, return_raw=True)
                    plt.plot(200 * plt_cx, 100 * plt_cy, c=color(k), linestyle="dotted", linewidth=2)
                for j in range(10):
                    s = np.array(c.sample(contexts[i:i + 1]))
                    spline = self._data.get_spline(s[0])
                    l, = plt.plot(200 * x_plt, 100 * spline(x_plt), c=color(k), linewidth=1)
                lines.append(l)
            for j in range(10):
                s = self._data.raw_test_samples[1][i, j]
                spline = self._data.get_spline(s)
                plt.plot(200 * x_plt, 100 * spline(x_plt), c=color(model.num_components), linewidth=1, linestyle="dashed")

            weights = model.gating_distribution.probabilities(context)[0]
            strs = ["{:.3f}".format(weights[i]) for i in range(model.num_components)]
            plt.legend(lines, strs, loc=1)
            plt.gca().set_axis_off()
            plt.gca().set_xlim(0, 200)
            plt.gca().set_ylim(0, 100)


class EMM1DModelRecMod(ModelRecModWithModelVis):
    """Visualization for Mixture of Experts with 1D input and output"""
    def __init__(self, data_obj, train_samples, test_samples, x_lim, true_log_density=None, eval_fn=None, test_log_iters=50,
                 save_log_iters=50):
        super().__init__(train_samples, test_samples, true_log_density, eval_fn, test_log_iters, save_log_iters)
        self._x_lim = x_lim
        self._data_obj = data_obj
        self._plt_x = \
            np.expand_dims(np.arange(self._x_lim[0], self._x_lim[1], (self._x_lim[1] - self._x_lim[0]) / 1000), -1)

    def _plot_model(self, emm, title):
        plt.subplot(3, 1, 1)
        plt.title(title)
        plt.scatter(np.squeeze(self._train_samples[0]), np.squeeze(self._train_samples[1]))
        axis = plt.gca()
        axis.set_xlim([self._x_lim[0], self._x_lim[1]])
        y_lim = axis.get_ylim()

        plt.subplot(3, 1, 2)
        for i in range(self._data_obj.num_modes):
            mean, cov = self._data_obj.get_conditional_parameters(self._plt_x, i)
            std = np.sqrt(cov[..., 0])
            plt.plot(self._plt_x, mean, c=self._colors(0))
            plt.fill_between(np.squeeze(self._plt_x), np.squeeze(mean) - 2 * std, np.squeeze(mean) + 2 * std,
                             alpha=0.5, edgecolor=self._colors(0), facecolor=self._colors(0))

        for i, c in enumerate(emm.components):
            mean = c.mean(self._plt_x)
            if callable(c.covar):
                std = np.sqrt(np.squeeze(c.covar(self._plt_x)))
            else:
                std = np.sqrt(np.squeeze(c.covar))
            plt.plot(self._plt_x, mean, c=self._colors(i+1))
            plt.fill_between(np.squeeze(self._plt_x), np.squeeze(mean) - 2 * std, np.squeeze(mean) + 2 * std,
                             alpha=0.5, edgecolor=self._colors(i+1), facecolor=self._colors(i+1))
        plt.gca().set_xlim([self._x_lim[0], self._x_lim[1]])
        plt.gca().set_ylim(y_lim)
        plt.grid(True)

        #plt.subplot(3, 1, 3)
        #weights = emm.weights(self._plt_x)
        #for i in range(len(emm.components)):
        #    plt.plot(self._plt_x, weights[:, i], c=self._colors(i+1))
        #plt.gca().set_xlim([self._x_lim[0], self._x_lim[1]])
        #plt.gca().set_ylim(-0.1, 1.1)
        plt.grid(True)
