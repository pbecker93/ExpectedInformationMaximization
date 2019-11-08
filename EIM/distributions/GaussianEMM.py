from distributions.Softmax import Softmax
from distributions.ConditionalGaussian import ConditionalGaussian
import numpy as np
import tensorflow as tf
import os
import shutil
import yaml


class GaussianEMM:

    def __init__(self, context_dim, sample_dim, number_of_components, component_hidden_dict, gating_hidden_dict,
                 seed=0, trainable=True, weight_path=None):

        self._context_dim = context_dim
        self._sample_dim = sample_dim
        self._number_of_components = number_of_components

        self._mixture_hidden_dict = gating_hidden_dict
        self._component_hidden_dict = component_hidden_dict

        wp = None if weight_path is None else os.path.join(weight_path, self._mixture_params_file_name())
        self._gating_distribution = Softmax(self._context_dim, self._number_of_components, gating_hidden_dict,
                                            seed=seed, trainable=trainable, weight_path=wp)

        self._components = []
        for i in range(number_of_components):
            h_dict = component_hidden_dict[i] if isinstance(component_hidden_dict, list) else component_hidden_dict
            wp = None if weight_path is None else os.path.join(weight_path, self._component_params_file_name(i))
            c = ConditionalGaussian(self._context_dim, self._sample_dim, h_dict,
                                    trainable=trainable, seed=seed, weight_path=wp)
            self._components.append(c)

        self.trainable_variables = self._gating_distribution.trainable_variables
        for c in self._components:
            self.trainable_variables += c.trainable_variables

    def density(self, contexts, samples):
        p = self._gating_distribution.probabilities(contexts)
        density = p[:, 0] * self._components[0].density(contexts, samples)
        for i in range(1, self._number_of_components):
            density += p[:, i] * self._components[i].density(contexts, samples)
        return density

    def log_density(self, contexts, samples):
        return tf.math.log(self.density(contexts, samples) + 1e-12)

    def log_likelihood(self, contexts, samples):
        return tf.reduce_mean(self.log_density(contexts, samples))

    def sample(self, contexts):
        modes = self._gating_distribution.sample(contexts)
        samples = np.zeros([contexts.shape[0], self._sample_dim])
        for i in range(self._number_of_components):
            idx = (modes == i)
            if np.any(idx):
                samples[idx] = self._components[i].sample(contexts[idx])
        return samples

    @property
    def gating_distribution(self):
        return self._gating_distribution

    @property
    def components(self):
        return self._components

    @property
    def num_components(self):
        return len(self._components)

    def get_component_parameters(self, context):
        means = []
        chol_covars = []
        for c in self._components:
            m, cc = c.conditional_params(context)
            means.append(m)
            chol_covars.append(cc)
        return tf.stack(means, 1), tf.stack(chol_covars, 1)

    def get_save_dict(self):
        return {
            "context_dim": self._context_dim,
            "x_dim": self._sample_dim,
            "number_of_components": self._number_of_components,
            "component_hidden_dict": self._component_hidden_dict,
            "mixture_hidden_dict": self._mixture_hidden_dict
        }

    @staticmethod
    def _component_params_file_name(i):
        return "weights_component_{}.h5".format(i)

    @staticmethod
    def _mixture_params_file_name():
        return "weights_mixture.h5"

    def save(self, path, filename):
        if not os.path.exists(path):
            os.mkdir(path)

        tmp_path = os.path.join(path, "tmp")
        os.mkdir(tmp_path)

        for i, c in enumerate(self.components):
            c.save_model_params(os.path.join(tmp_path, self._component_params_file_name(i)))
        self._gating_distribution.save_model_params(os.path.join(tmp_path, self._mixture_params_file_name()))
        with open(os.path.join(tmp_path, "config_dict.json"), "w") as f:
            f.write(yaml.dump(self.get_save_dict()))

        shutil.make_archive(os.path.join(path, filename), "tar", tmp_path)
        shutil.rmtree(tmp_path)

    @staticmethod
    def from_file(path, filename, seed=0, trainable=True):
        if filename[:-4] is not ".tar":
            filename += ".tar"
        tmp_path = os.path.join(path, "tmp")
        os.mkdir(tmp_path)
        shutil.unpack_archive(os.path.join(path, filename), tmp_path, "tar")
        with open(os.path.join(tmp_path, "config_dict.json"), "r") as f:
            config_dict = yaml.load(f)

        model = GaussianEMM(context_dim=config_dict["context_dim"], sample_dim=config_dict["x_dim"],
                            number_of_components=config_dict["number_of_components"],
                            component_hidden_dict=config_dict["component_hidden_dict"],
                            gating_hidden_dict=config_dict["mixture_hidden_dict"],
                            seed=seed, trainable=trainable, weight_path=tmp_path)

        shutil.rmtree(tmp_path)
        return model
