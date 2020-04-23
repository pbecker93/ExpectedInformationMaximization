import tensorflow as tf
from tensorflow import keras as k
import util.NetworkBuilder as nb
import numpy as np
import os
import shutil
import yaml


def gaussian_log_density(samples, means, chol_covars):
    covar_logdet = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_covars)), -1)
    diff = tf.expand_dims(samples - means, -1)
    exp_term = tf.reduce_sum(tf.square(tf.linalg.triangular_solve(chol_covars, diff)), axis=[-2, -1])
    return - 0.5 * (tf.cast(tf.shape(samples)[-1], exp_term.dtype) * np.log(2 * np.pi) + covar_logdet + exp_term)

def gaussian_density(samples, means, chol_covars):
    return tf.exp(ConditionalGaussian.gaussian_log_density(samples, means, chol_covars))


class ConditionalGaussian:

    def __init__(self, context_dim, sample_dim, hidden_dict, seed, trainable=True, weight_path=None):
        self._context_dim = context_dim
        self._sample_dim = sample_dim
        self._seed = seed
        self._hidden_dict = hidden_dict

        self.trainable = trainable

        self._build(weight_path)

    def _build(self, weight_path):
        self._hidden_net = nb.build_dense_network(self._context_dim, output_dim=-1, output_activation=None,
                                                  params=self._hidden_dict, with_output_layer=False)

        self._mean_t = k.layers.Dense(self._sample_dim)(self._hidden_net.output)
        self._chol_covar_raw = k.layers.Dense(self._sample_dim ** 2)(self._hidden_net.output)
        self._chol_covar = k.layers.Lambda(self._create_chol)(self._chol_covar_raw)

        self._cond_params_model = k.models.Model(inputs=[self._hidden_net.inputs],
                                                 outputs=[self._mean_t, self._chol_covar])

        if weight_path is not None:
            self._cond_params_model.load_weights(weight_path)

    def mean(self, contexts):
        return self._cond_params_model(contexts)[0]

    def covar(self, contexts):
        chol_covar = self._cond_params_model(contexts)[1]
        return tf.matmul(chol_covar, chol_covar, transpose_b=True)

    def _create_chol(self, chol_raw):
        samples = tf.linalg.band_part(tf.reshape(chol_raw, [-1, self._sample_dim, self._sample_dim]), -1, 0)
        return tf.linalg.set_diag(samples, tf.exp(tf.linalg.diag_part(samples)) + 1e-12)

    def sample(self, contexts):
        mean, chol_covar = self._cond_params_model(contexts)
        eps = tf.random.normal([tf.shape(mean)[0], tf.shape(mean)[1], 1], seed=self._seed)
        return mean + tf.reshape(tf.matmul(chol_covar, eps), tf.shape(mean))

    def log_density(self, contexts, samples):
        mean, chol_covar = self._cond_params_model(contexts)
        return gaussian_log_density(samples, mean, chol_covar)

    def density(self, contexts, samples):
        return tf.exp(self.log_density(contexts, samples))

    def expected_entropy(self, contexts):
        _, chol_covars = self._cond_params_model(contexts)
        return 0.5 * (self._sample_dim * np.log(2 * np.e * np.pi) + tf.reduce_mean(self._covar_logdets(chol_covars)))

    def entropies(self, contexts):
        _, chol_covars = self._cond_params_model(contexts)
        return 0.5 * (self._sample_dim * np.log(2 * np.e * np.pi) + self._covar_logdets(chol_covars))

    def _covar_logdets(self, chol_covars):
        return 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_covars) + 1e-12), -1)

    def kls(self, contexts, other_means, other_chol_covars):
        means, chol_covars = self._cond_params_model(contexts)
        kl = self._covar_logdets(other_chol_covars) - self._covar_logdets(chol_covars) - self._sample_dim
        kl += tf.reduce_sum(tf.square(tf.linalg.triangular_solve(other_chol_covars, chol_covars)), axis=[-2, -1])
        diff = tf.expand_dims(other_means - means, -1)
        kl += tf.reduce_sum(tf.square(tf.linalg.triangular_solve(other_chol_covars, diff)), axis=[-2, -1])
        return 0.5 * kl

    def kls_other_chol_inv(self, contexts, other_means, other_chol_inv):
        means, chol_covars = self._cond_params_model(contexts)
        kl = - self._covar_logdets(other_chol_inv) - self._covar_logdets(chol_covars) - self._sample_dim
        kl += tf.reduce_sum(tf.square(tf.matmul(other_chol_inv, chol_covars)), axis=[-2, -1])
        diff = tf.expand_dims(other_means - means, -1)
        kl += tf.reduce_sum(tf.square(tf.matmul(other_chol_inv, diff)), axis=[-2, -1])
        return 0.5 * kl

    def expected_kl(self, contexts, other_means, other_chol_covars):
        return tf.reduce_mean(self.kls(contexts, other_means, other_chol_covars))

    def log_likelihood(self, contexts, samples):
        return tf.reduce_mean(self.log_density(contexts, samples))

    def conditional_params(self, contexts):
        return self._cond_params_model(contexts)

    @property
    def trainable_variables(self):
        if self.trainable:
            return self._cond_params_model.trainable_variables
        else:
            return []

    @property
    def sample_dim(self):
        return self._sample_dim

    def save_model_params(self, filepath):
        self._cond_params_model.save_weights(filepath=filepath, overwrite=False)

    def get_save_dict(self):
        return {"context_dim": self._context_dim,
                "sample_dim": self._sample_dim,
                "hidden_dict": self._hidden_dict}

    def save(self, path, filename):
        if not os.path.exists(path):
            os.mkdir(path)

        tmp_path = os.path.join(path, "tmp")
        os.mkdir(tmp_path)

        self.save_model_params(os.path.join(tmp_path, "weights.h5"))
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
        model = ConditionalGaussian(context_dim=config_dict["context_dim"], sample_dim=config_dict["sample_dim"],
                                    hidden_dict=config_dict["hidden_dict"], seed=seed, trainable=trainable,
                                    weight_path=os.path.join(tmp_path, "weights.h5"))
        shutil.rmtree(tmp_path)
        return model
