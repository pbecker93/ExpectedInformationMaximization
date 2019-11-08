from distributions.GaussianEMM import GaussianEMM
from recording.Recorder import Recorder
import tensorflow as tf
from tensorflow import keras as k
from util.ConfigDict import ConfigDict
import numpy as np
from recording.Recorder import RecorderKeys as rec
from eim.DensityRatioEstimator import DensityRatioEstimator
from util.NetworkBuilder import NetworkKeys
import logging

class ConditionalMixtureEIM:

    @staticmethod
    def get_default_config() -> ConfigDict:
        c = ConfigDict(
            num_components=1,
            train_epochs=1000,
            # Component
            components_learning_rate=1e-3,
            components_batch_size=1000,
            components_num_epochs=10,
            components_net_reg_loss_fact=0.,
            components_net_drop_prob=0.0,
            components_net_hidden_layers=[50, 50],
            # Gating
            gating_learning_rate=1e-3,
            gating_batch_size=1000,
            gating_num_epochs=10,
            gating_net_reg_loss_fact=0.,
            gating_net_drop_prob=0.0,
            gating_net_hidden_layers=[50, 50],
            # Density Ratio Estimation
            dre_reg_loss_fact=0.0,  # Scaling Factor for L2 regularization of density ratio estimator
            dre_early_stopping=True,  # Use early stopping for density ratio estimator training
            dre_drop_prob=0.0,  # If smaller than 1 dropout with keep prob = 'keep_prob' is used
            dre_num_iters=1000,  # Number of density ratio estimator steps each iteration (i.e. max number if early stopping)
            dre_batch_size=1000,  # Batch size for density ratio estimator training
            dre_hidden_layers=[30, 30]  # width of density ratio estimator  hidden layers
        )
        c.finalize_adding()
        return c

    def __init__(self, config: ConfigDict, train_samples: np.ndarray, recorder: Recorder,
                 val_samples: np.ndarray = None, seed: int = 0):
        # supress tensorflow casting warnings
        logging.getLogger("tensorflow").setLevel(logging.ERROR)

        self.c = config
        self.c.finalize_modifying()

        self._recorder = recorder

        self._context_dim = train_samples[0].shape[-1]
        self._sample_dim = train_samples[1].shape[-1]
        self._train_contexts = tf.constant(train_samples[0], dtype=tf.float32)

        c_net_hidden_dict = {NetworkKeys.NUM_UNITS: self.c.components_net_hidden_layers,
                             NetworkKeys.ACTIVATION: "relu",
                             NetworkKeys.BATCH_NORM: False,
                             NetworkKeys.DROP_PROB: self.c.components_net_drop_prob,
                             NetworkKeys.L2_REG_FACT: self.c.components_net_reg_loss_fact}

        g_net_hidden_dict = {NetworkKeys.NUM_UNITS: self.c.gating_net_hidden_layers,
                             NetworkKeys.ACTIVATION: "relu",
                             NetworkKeys.BATCH_NORM: False,
                             NetworkKeys.DROP_PROB: self.c.gating_net_drop_prob,
                             NetworkKeys.L2_REG_FACT: self.c.gating_net_reg_loss_fact}

        self._model = GaussianEMM(self._context_dim, self._sample_dim, self.c.num_components,
                                  c_net_hidden_dict, g_net_hidden_dict, seed=seed)
        self._c_opts = [k.optimizers.Adam(self.c.components_learning_rate, 0.5) for _ in self._model.components]
        self._g_opt = k.optimizers.Adam(self.c.gating_learning_rate, 0.5)

        dre_params = {NetworkKeys.NUM_UNITS: self.c.dre_hidden_layers,
                      NetworkKeys.ACTIVATION: k.activations.relu,
                      NetworkKeys.DROP_PROB: self.c.dre_drop_prob,
                      NetworkKeys.L2_REG_FACT: self.c.dre_reg_loss_fact}
        self._dre = DensityRatioEstimator(target_train_samples=train_samples,
                                          hidden_params=dre_params,
                                          early_stopping=self.c.dre_early_stopping, target_val_samples=val_samples,
                                          conditional_model=True)

        self._recorder.initialize_module(rec.INITIAL)
        self._recorder(rec.INITIAL, "Nonlinear Conditional EIM - Reparametrization", config)
        self._recorder.initialize_module(rec.MODEL, self.c.train_epochs)
        self._recorder.initialize_module(rec.WEIGHTS_UPDATE, self.c.train_epochs)
        self._recorder.initialize_module(rec.COMPONENT_UPDATE, self.c.train_epochs, self.c.num_components)
        self._recorder.initialize_module(rec.DRE, self.c.train_epochs)

    def train(self):
        for i in range(self.c.train_epochs):
            self._recorder(rec.TRAIN_ITER, i)
            self.train_iter(i)

    #   extra function to allow running from cluster work
    def train_iter(self, i):

        dre_steps, loss, acc = self._dre.train(self._model, self.c.dre_batch_size, self.c.dre_num_iters)
        self._recorder(rec.DRE, self._dre, self._model, i, dre_steps)

        if self._model.num_components > 1:
            w_res = self.update_gating()
            self._recorder(rec.WEIGHTS_UPDATE, w_res)

        c_res = self.update_components()
        self._recorder(rec.COMPONENT_UPDATE, c_res)

        self._recorder(rec.MODEL, self._model, i)

    """component update"""
    def update_components(self):
        importance_weights = self._model.gating_distribution.probabilities(self._train_contexts)
        importance_weights = importance_weights / tf.reduce_sum(importance_weights, axis=0, keepdims=True)

        old_means, old_chol_covars = self._model.get_component_parameters(self._train_contexts)

        rhs = tf.eye(tf.shape(old_means)[-1], batch_shape=tf.shape(old_chol_covars)[:-2])
        stab_fact = 1e-20
        old_chol_inv = tf.linalg.triangular_solve(old_chol_covars + stab_fact * rhs, rhs)

        for i in range(self.c.components_num_epochs):
            self._components_train_step(importance_weights, old_means, old_chol_inv)

        res_list = []
        for i, c in enumerate(self._model.components):
            expected_entropy = np.sum(importance_weights[:, i] * c.entropies(self._train_contexts))
            kls = c.kls_other_chol_inv(self._train_contexts, old_means[:, i], old_chol_inv[:, i])
            expected_kl = np.sum(importance_weights[:, i] * kls)
            res_list.append((self.c.components_num_epochs, expected_kl, expected_entropy, ""))
        return res_list

    @tf.function
    def _components_train_step(self, importance_weights, old_means, old_chol_precisions):
        for i in range(self._model.num_components):
            dt = (self._train_contexts, importance_weights[:, i], old_means, old_chol_precisions)
            data = tf.data.Dataset.from_tensor_slices(dt)
            data = data.shuffle(self._train_contexts.shape[0]).batch(self.c.components_batch_size)

            for context_batch, iw_batch, old_means_batch, old_chol_precisions_batch in data:
                iw_batch = iw_batch / tf.reduce_sum(iw_batch)
                with tf.GradientTape() as tape:
                    samples = self._model.components[i].sample(context_batch)
                    losses = - tf.squeeze(self._dre(tf.concat([context_batch, samples], axis=-1)))
                    kls = self._model.components[i].kls_other_chol_inv(context_batch, old_means_batch[:, i],
                                                                  old_chol_precisions_batch[:, i])
                    loss = tf.reduce_mean(iw_batch * (losses + kls))
                gradients = tape.gradient(loss, self._model.components[i].trainable_variables)
                self._c_opts[i].apply_gradients(zip(gradients, self._model.components[i].trainable_variables))


    """gating update"""
    def update_gating(self):
        old_probs = self._model.gating_distribution.probabilities(self._train_contexts)
        for i in range(self.c.gating_num_epochs):
            self._gating_train_step(old_probs)

        expected_entropy = self._model.gating_distribution.expected_entropy(self._train_contexts)
        expected_kl = self._model.gating_distribution.expected_kl(self._train_contexts, old_probs)

        return i + 1, expected_kl, expected_entropy, ""

    @tf.function
    def _gating_train_step(self, old_probs):
        losses = []
        for i in range(self.c.num_components):
            samples = self._model.components[i].sample(self._train_contexts)
            losses.append(- self._dre(tf.concat([self._train_contexts, samples], axis=-1)))

        losses = tf.concat(losses, axis=1)
        data = tf.data.Dataset.from_tensor_slices((self._train_contexts, losses, old_probs))
        data = data.shuffle(self._train_contexts.shape[0]).batch(self.c.gating_batch_size)
        for context_batch, losses_batch, old_probs_batch in data:
            with tf.GradientTape() as tape:
                probabilities = self._model.gating_distribution.probabilities(context_batch)
                kl = self._model.gating_distribution.expected_kl(context_batch, old_probs_batch)
                loss = tf.reduce_sum(tf.reduce_mean(probabilities * losses_batch, 0)) + kl
            gradients = tape.gradient(loss, self._model.gating_distribution.trainable_variables)
            self._g_opt.apply_gradients(zip(gradients, self._model.gating_distribution.trainable_variables))

    @property
    def model(self):
        return self._model
