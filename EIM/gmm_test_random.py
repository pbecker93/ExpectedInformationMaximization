
from recording.Recorder import RecorderKeys as rec_keys
from recording.Recorder import Recorder
from recording.modules.SimpleModules import TrainIterationRecMod, ConfigInitialRecMod
from recording.modules.ModelModules import ModelRecMod
from recording.modules.DREModules import DRERecMod
from recording.modules.UpdateModules import WeightUpdateRecMod, ComponentUpdateRecMod
from data.RandomGMMData import RandomGMMData
import numpy as np
from eim.MarginalMixtureEIM import MarginalMixtureEIM

"""Runs EIM for data generated from randomly generated GMMs"""

dim = 5
num_components = 10


"""Create data target distribution"""
data = RandomGMMData(dimensionality=dim, num_components=num_components,
                     mean_limits=[-1, 1], covar_variance=np.sqrt(2) * 0.05, seed=42,
                     num_train_samples=10000, num_test_samples=5000, num_val_samples=2000)
""" Recording """
recorder_dict = {
    rec_keys.TRAIN_ITER: TrainIterationRecMod(),
    rec_keys.INITIAL: ConfigInitialRecMod(),
    rec_keys.MODEL: ModelRecMod(test_log_iters=50,
                                save_log_iters=10,
                                eval_fn=None,
                                train_samples=data.train_samples,
                                test_samples=data.test_samples,
                                true_log_density=data.log_density),
    rec_keys.WEIGHTS_UPDATE: WeightUpdateRecMod(plot=True),
    rec_keys.COMPONENT_UPDATE: ComponentUpdateRecMod(plot=True, summarize=True),
    rec_keys.DRE: DRERecMod(data.train_samples)
}

# Set 'plot_realtime' to True for some nice visualization
recorder = Recorder(recorder_dict, plot_realtime=True, save=False)

"""Running the algorithm """

config = MarginalMixtureEIM.get_default_config()
config.num_components = num_components
config.train_epochs = 1000
config.component_kl_bound = 0.05
config.weight_kl_bound = 0.05
config.initialization = "random"
config.dre_hidden_layers = [50, 50, 50]
config.dre_reg_loss_fact = 0.001

model = MarginalMixtureEIM(config=config, train_samples=data.train_samples, recorder=recorder, seed=0,
                           val_samples=data.val_samples)
model.train()

