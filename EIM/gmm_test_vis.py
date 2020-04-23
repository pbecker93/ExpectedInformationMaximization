from recording.Recorder import RecorderKeys as rec_keys
from recording.Recorder import Recorder
from recording.modules.SimpleModules import TrainIterationRecMod, ConfigInitialRecMod
from recording.modules.ModelModules import GMM2DModelRecMod
from recording.modules.DREModules import DRERecMod
from recording.modules.UpdateModules import WeightUpdateRecMod, ComponentUpdateRecMod
from distributions.marginal.GMM import GMM

import numpy as np
from data.RandomGMMData import GMMData
from eim.MarginalMixtureEIM import MarginalMixtureEIM

"""Simple Test Script for fitting 2D GMMs with EIM. This does not correspond to any specific experiment in the paper
and is mostly for demonstrative and visualization purposes"""

"""Create some target distribution"""
target_weights = np.array([0.2, 0.4, 0.4])
target_means = np.array([[0.0, 0.0], [-2.0, 2.0], [2.0, 2.0]])
#                                    c1                        c2                        c3
target_covars = np.array([[[0.7, 0.05], [0.05, 0.1]], [[0.1, -0.02], [-0.02, 0.7]], [[0.1, -0.09], [-0.09, 0.7]]])

target_distribution = GMM(target_weights, target_means, target_covars)

data = GMMData(target_weights, target_means, target_covars, num_train_samples=10000, num_test_samples=5000,
               num_val_samples=2000, seed=0)

""" Recording """
recorder_dict = {
    rec_keys.TRAIN_ITER: TrainIterationRecMod(),
    rec_keys.INITIAL: ConfigInitialRecMod(),
    rec_keys.MODEL: GMM2DModelRecMod(test_log_iters=50,
                                     save_log_iters=10,
                                     eval_fn=None,
                                     train_samples=data.train_samples,
                                     test_samples=data.test_samples,
                                     true_means=target_means,
                                     true_covars=target_covars,
                                     true_log_density=target_distribution.log_density),
    rec_keys.WEIGHTS_UPDATE: WeightUpdateRecMod(plot=True),
    rec_keys.COMPONENT_UPDATE: ComponentUpdateRecMod(plot=True, summarize=False),
    rec_keys.DRE: DRERecMod(data.train_samples)
}

# Set 'plot_realtime' to True for some nice visualization
recorder = Recorder(recorder_dict, plot_realtime=True, save=False)

"""Running the algorithm """

config = MarginalMixtureEIM.get_default_config()
config.num_components = 3
config.train_epochs = 250
config.component_kl_bound = 0.05
config.weight_kl_bound = 0.05
config.initialization = "random"
config.dre_hidden_layers = [50, 50, 50]
model = MarginalMixtureEIM(config=config, train_samples=data.train_samples, recorder=recorder, seed=0,
                           val_samples=data.val_samples)
model.train()

