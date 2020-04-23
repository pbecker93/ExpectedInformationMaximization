from tensorflow import keras as k
import numpy as np
from util.ConfigDict import ConfigDict
from recording.Recorder import RecorderKeys as rec
from eim.DensityRatioEstimator import DensityRatioEstimator, AddFeatDensityRatioEstimator
from util.NetworkBuilder import NetworkKeys
import util.ModelInit as model_init
from itps.MoreGaussian import MoreGaussian
from itps.RepsCategorical import RepsCategorical
from distributions.marginal.GMM import GMM
from distributions.marginal.Categorical import Categorical
from distributions.marginal.Gaussian import Gaussian
from util.Regression import QuadFunc


class MarginalMixtureEIM:

    @staticmethod
    def get_default_config():
        c = ConfigDict(
            num_components=1,
            samples_per_component=500,
            train_epochs=1000,
            initialization="random",
            # Component Updates
            component_kl_bound=0.01,
            # Mixture Updates
            weight_kl_bound=0.01,
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

    def __init__(self, config, train_samples, recorder, val_samples=None, seed=0, add_feat_fn=None):

        self.c = config
        self.c.finalize_modifying()

        # build model
        w, m, c = model_init.gmm_init(self.c.initialization, np.array(train_samples, dtype=np.float32),
                                      self.c.num_components, seed=0)
        self._model = GMM(w, m, c)

        self._components_learners = []
        for i in range(self._model.num_components):
            self._components_learners.append(MoreGaussian(m.shape[-1], 1.0, 0.0, False))
        self._weight_learner = RepsCategorical(1.0, 0.0, False)

        # build density ratio estimator
        dre_params = {NetworkKeys.NUM_UNITS: self.c.dre_hidden_layers,
                      NetworkKeys.ACTIVATION: k.activations.relu,
                      NetworkKeys.DROP_PROB: self.c.dre_drop_prob,
                      NetworkKeys.L2_REG_FACT: self.c.dre_reg_loss_fact}
        if add_feat_fn is not None:
            self._dre = AddFeatDensityRatioEstimator(target_train_samples=train_samples, hidden_params=dre_params,
                                                     early_stopping=self.c.dre_early_stopping,
                                                     target_val_samples=val_samples, additional_feature_fn=add_feat_fn)
            name = "Marginal EIM - Additional Features"
        else:
            self._dre = DensityRatioEstimator(target_train_samples=train_samples, hidden_params=dre_params,
                                              early_stopping=self.c.dre_early_stopping, target_val_samples=val_samples)
            name = "Marginal EIM"

        # build recording
        self._recorder = recorder
        self._recorder.initialize_module(rec.INITIAL)
        self._recorder(rec.INITIAL, name, config)
        self._recorder.initialize_module(rec.MODEL, self.c.train_epochs)
        self._recorder.initialize_module(rec.WEIGHTS_UPDATE, self.c.train_epochs)
        self._recorder.initialize_module(rec.COMPONENT_UPDATE, self.c.train_epochs, self.c.num_components)
        self._recorder.initialize_module(rec.DRE, self.c.train_epochs)

    @property
    def model(self):
        return self._model

    def train(self):
        for i in range(self.c.train_epochs):
            self._recorder(rec.TRAIN_ITER, i)

            # update density ratio estimator
            dre_steps, loss, acc = self._dre.train(self.model, self.c.dre_batch_size, self.c.dre_num_iters)
            self._recorder(rec.DRE, self._dre, self.model, i, dre_steps)

            # M-Step weights
            if self.model.num_components > 1:
                w_res = self.update_weight()
                self._recorder(rec.WEIGHTS_UPDATE, w_res)

            # M-Step components
            c_res = self.update_components()
            self._recorder(rec.COMPONENT_UPDATE, c_res)

            self._recorder(rec.MODEL, self.model, i)

    def _get_reward(self, samples_as_list):
        rewards = []
        for i, samples in enumerate(samples_as_list):
            rewards.append(self._dre(samples))
        return rewards

    def _get_samples(self):
        samples = []
        for c in self.model.components:
            samples.append(c.sample(self.c.samples_per_component))
        return samples

    def update_components(self):
        samples = self._get_samples()
        rewards = self._get_reward(samples)

        res_list = []

        for i in range(self._model.num_components):
            component = self._model.components[i]
            learner = self._components_learners[i]

            old_dist = Gaussian(component.mean, component.covar)

            surrogate = QuadFunc(1e-12, normalize=True, unnormalize_output=False)
            surrogate.fit(samples[i], rewards[i], None, old_dist.mean, old_dist.chol_covar)

            # This is a numerical thing we did not use in the original paper: We do not undo the output normalization
            # of the regression, this will yield the same solution but the optimal lagrangian multipliers of the
            # MORE dual are scaled, so we also need to adapt the offset. This makes optimizing the dual much more
            # stable and indifferent to initialization
            learner.eta_offset = 1.0 / surrogate.o_std

            new_mean, new_covar = learner.more_step(self.c.component_kl_bound, -1, component, surrogate)
            if learner.success:
                component.update_parameters(new_mean, new_covar)
                res_list.append((1, component.kl(old_dist), component.entropy(), " "))
            else:
                res_list.append((1, 0.0, old_dist.entropy(), "update of component {:d} failed".format(i)))

        return res_list

    def update_weight(self):
        fake_samples = self._get_samples()
        rewards = np.mean(self._get_reward(fake_samples), 1)
        rewards = np.ascontiguousarray(np.concatenate(rewards, -1).astype(np.float64))

        old_dist = Categorical(self._model.weight_distribution.probabilities)

        # -1 as entropy bound is a dummy as entropy is not constraint
        new_probabilities = self._weight_learner.reps_step(self.c.weight_kl_bound, -1, old_dist, rewards)
        if self._weight_learner.success:
            self._model.weight_distribution.probabilities = new_probabilities

        kl = self._model.weight_distribution.kl(old_dist)
        entropy = self._model.weight_distribution.entropy()
        return 1, kl, entropy, " "


