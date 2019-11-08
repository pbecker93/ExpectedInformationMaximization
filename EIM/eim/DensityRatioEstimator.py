from tensorflow import keras as k
from util.NetworkBuilder import build_dense_network
import tensorflow as tf
import numpy as np


def accuracy(p_outputs, q_outputs):
    p_prob = tf.sigmoid(p_outputs)
    q_prob = tf.sigmoid(q_outputs)
    return tf.reduce_mean(tf.cast(tf.concat([tf.greater_equal(p_prob, 0.5), tf.less(q_prob, 0.5)], 0), tf.float32))


def logistic_regression_loss(p_outputs, q_outputs):
    return - tf.reduce_mean(tf.math.log(tf.sigmoid(p_outputs) + 1e-12)) \
           - tf.reduce_mean(tf.math.log(1 - tf.sigmoid(q_outputs) + 1e-12))


class DensityRatioEstimator:

    def __init__(self, target_train_samples, hidden_params, early_stopping=False, target_val_samples=None,
                 conditional_model=False):

        self._early_stopping = early_stopping
        self._conditional_model = conditional_model

        if self._conditional_model:
            self._train_contexts = target_train_samples[0].astype(np.float32)
            self._target_train_samples = np.concatenate(target_train_samples, -1).astype(np.float32)
        else:
            self._target_train_samples = target_train_samples.astype(np.float32)

        if self._early_stopping:
            assert target_val_samples is not None, \
                "For early stopping validation data needs to be provided via target_val_samples"
            if self._conditional_model:
                self._val_contexts = target_val_samples[0].astype(np.float32)
                self._target_val_samples = np.concatenate(target_val_samples, -1).astype(np.float32)
            else:
                self._target_val_samples = target_val_samples.astype(np.float32)

        input_dim = self._target_train_samples.shape[-1]

        self._ldre_net = build_dense_network(input_dim=input_dim, output_dim=1,
                                             output_activation=k.activations.linear, params=hidden_params)

        self._p_samples = k.layers.Input(shape=(input_dim,))
        self._q_samples = k.layers.Input(shape=(input_dim,))

        self._train_model = k.models.Model(inputs=[self._p_samples, self._q_samples],
                                           outputs=[self._ldre_net(self._p_samples), self._ldre_net(self._q_samples)])

        self._train_model.add_loss(logistic_regression_loss(self._train_model.outputs[0], self._train_model.outputs[1]))
        self._acc = accuracy(self._train_model.outputs[0], self._train_model.outputs[1])
        self._train_model.add_metric(self._acc, name="acc", aggregation="mean")
        self._parameters = None

        self._train_model.compile(optimizer=k.optimizers.Adam())

    def __call__(self, samples):
        return self._ldre_net(samples)

    def train(self, model, batch_size, num_iters):
        if self._early_stopping:
            model_train_samples, model_val_samples = self.sample_model(model)
            callbacks = [k.callbacks.EarlyStopping()]
            validation_data = ((self._target_val_samples, model_val_samples), None)
        else:
            model_train_samples = self.sample_model(model)
            callbacks = []
            validation_data = None
        hist = self._train_model.fit(x=(self._target_train_samples, model_train_samples),
                                     batch_size=batch_size,
                                     epochs=num_iters,
                                     verbose=0,
                                     validation_data=validation_data,
                                     callbacks=callbacks)

        last_epoch = hist.epoch[-1]
        return last_epoch + 1, hist.history["loss"][last_epoch], hist.history["acc"][last_epoch]

    def eval(self, target_samples, model):
        if self._conditional_model:
            model_samples = np.concatenate([target_samples[0], model.sample(target_samples[0])], axis=-1)
            target_samples = np.concatenate(target_samples, axis=-1)
        else:
            model_samples = model.sample(self._target_train_samples.shape[0])
        target_ldre = self(target_samples)
        model_ldre = self(model_samples)
        target_prob = tf.nn.sigmoid(target_ldre)
        model_prob = tf.nn.sigmoid(model_ldre)

        ikl_estem = tf.reduce_mean(- model_ldre)
        acc = accuracy(target_ldre, model_ldre)
        bce = logistic_regression_loss(target_ldre, model_ldre)
        return ikl_estem, bce, acc, tf.reduce_mean(target_prob), tf.reduce_mean(model_prob)

    def sample_model(self, model):
        """Sample model for density ratio estimator training"""
        if self._conditional_model:
            model_train_samples = np.concatenate([self._train_contexts, model.sample(self._train_contexts)], axis=-1)
        else:
            model_train_samples = model.sample(self._target_train_samples.shape[0])
        if self._early_stopping:
            if self._conditional_model:
                model_val_samples = np.concatenate([self._val_contexts, model.sample(self._val_contexts)], axis=-1)
            else:
                model_val_samples = model.sample(self._target_val_samples.shape[0])
            return model_train_samples, model_val_samples
        return model_train_samples


class AddFeatDensityRatioEstimator(DensityRatioEstimator):

    def __init__(self, target_train_samples, hidden_params, additional_feature_fn,
                 early_stopping=False, target_val_samples=None):

        self._additional_feature_fn = additional_feature_fn

        target_train_add_feat = self._additional_feature_fn(target_train_samples)
        target_train_samples = np.concatenate([target_train_samples, target_train_add_feat], axis=-1)

        if target_val_samples is not None:
            target_val_add_feat = self._additional_feature_fn(target_val_samples)
            target_val_samples = np.concatenate([target_val_samples, target_val_add_feat], axis=-1)

        super().__init__(target_train_samples, hidden_params, early_stopping, target_val_samples)

    def __call__(self, samples, *args):
        add_features = self._additional_feature_fn(samples)
        return self._ldre_net(np.concatenate([samples, add_features], axis=-1))

    def sample_model(self, model):
        """Sample model for density ratio estimator training - add additional features"""
        model_train_samples = model.sample(self._target_train_samples.shape[0])
        model_train_add_feat = self._additional_feature_fn(model_train_samples)
        model_train_samples = np.concatenate([model_train_samples, model_train_add_feat], axis=-1)
        if self._early_stopping:
            model_val_samples = model.sample(self._target_val_samples.shape[0])
            model_add_feat = self._additional_feature_fn(model_val_samples)
            model_val_samples = np.concatenate([model_val_samples, model_add_feat], axis=-1)
            return model_train_samples, model_val_samples
        return model_train_samples
