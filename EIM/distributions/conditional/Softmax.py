import tensorflow as tf
import util.NetworkBuilder as nb
from tensorflow import keras as k
import os
import shutil
import yaml


class Softmax:

    def __init__(self, context_dim, z_dim, hidden_dict, seed, trainable=True, weight_path=None):
        self._context_dim = context_dim
        self._z_dim = z_dim
        self._seed = seed
        self._hidden_dict = hidden_dict
        self._trainable = trainable

        self._logit_net = nb.build_dense_network(self._context_dim, self._z_dim, output_activation=k.activations.linear,
                                                 params=self._hidden_dict)

        if weight_path is not None:
            self._logit_net.load_weights(weight_path)

    def logits(self, contexts):
        return self._logit_net(contexts)

    def probabilities(self, contexts):
        return k.activations.softmax(self.logits(contexts))

    def log_probabilities(self, contexts):
        return tf.math.log(self.probabilities(contexts) + 1e-12)

    def expected_entropy(self, contexts):
        p = self.probabilities(contexts)
        return - tf.reduce_mean(tf.reduce_sum(p * tf.math.log(p + 1e-12), -1))

    def expected_kl(self, contexts, other_probabilities):
        p = self.probabilities(contexts)
        return \
            tf.reduce_mean(tf.reduce_sum(p * (tf.math.log(p + 1e-12) - tf.math.log(other_probabilities + 1e-12)), -1))

    def sample(self, contexts):
        p = self.probabilities(contexts)
        thresholds = tf.cumsum(p, axis=-1)
        # ensure the last threshold is always exactly one - it can be slightly smaller due to numerical inaccuracies
        # of cumsum, causing problems in extremely rare cases if a "n" is samples that's larger than the last threshold
        thresholds = tf.concat([thresholds[..., :-1], tf.ones([tf.shape(thresholds)[0], 1])], -1)
        n = tf.random.uniform(shape=[tf.shape(thresholds)[0], 1], minval=0.0, maxval=1.0, seed=self._seed)
        idx = tf.where(tf.less(n, thresholds), tf.range(self._z_dim) * tf.ones(thresholds.shape, dtype=tf.int32),
                       self._z_dim * tf.ones(thresholds.shape, dtype=tf.int32))
        return tf.reduce_min(idx, -1)

    @property
    def trainable_variables(self):
        return self._logit_net.trainable_variables

    def save_model_params(self, filepath):
        self._logit_net.save_weights(filepath=filepath, overwrite=False)

    def get_save_dict(self):
        return {"context_dim": self._context_dim,
                "z_dim": self._z_dim,
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
        model = Softmax(context_dim=config_dict["context_dim"], z_dim=config_dict["z_dim"],
                        hidden_dict=config_dict["hidden_dict"], seed=seed, trainable=trainable,
                        weight_path=os.path.join(tmp_path, "weights.h5"))
        shutil.rmtree(tmp_path)
        return model
