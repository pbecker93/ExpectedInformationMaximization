from recording.modules.ModelModules import ObstacleModelRecMod
from recording.Recorder import RecorderKeys as rec_keys
from recording.Recorder import Recorder
from recording.modules.DREModules import DRERecMod
from recording.modules.UpdateModules import WeightUpdateRecMod, ComponentUpdateRecMod
from recording.modules.SimpleModules import ConfigInitialRecMod, TrainIterationRecMod
from data.ObstacleData import ObstacleData
from eim.ConditionalMixtureEIM import ConditionalMixtureEIM
import numpy as np

"""Runs the obstacles experiments described in section 5.4 of the paper"""

"""configure experiment"""
plot_realtime = True
plot_save = False
record_dual_opt = True
record_discriminator = True
num_components = 3

"""generate data"""

data = ObstacleData(10000, 5000, 5000, num_obstacles=3, samples_per_context=10, seed=0)
context_dim, sample_dim = data.dim

"""eval and recording"""


def eval_fn(model):
    contexts = data.raw_test_samples[0]
    samples = np.zeros([contexts.shape[0], 10, data.dim[1]])
    for i in range(10):
        samples[:, i] = model.sample(contexts)
    return data.rewards_from_contexts(contexts, samples)


"""Recording"""
recorder_dict = {
    rec_keys.TRAIN_ITER: TrainIterationRecMod(),
    rec_keys.INITIAL: ConfigInitialRecMod(),
    rec_keys.MODEL: ObstacleModelRecMod(data,
                                        train_samples=data.train_samples,
                                        test_samples=data.test_samples,
                                        test_log_iters=1,
                                        eval_fn=eval_fn,
                                        save_log_iters=50),
    rec_keys.DRE: DRERecMod(data.train_samples),
    rec_keys.COMPONENT_UPDATE: ComponentUpdateRecMod(plot=True, summarize=False)}
if num_components > 1:
    recorder_dict[rec_keys.WEIGHTS_UPDATE] = WeightUpdateRecMod(plot=True)


recorder = Recorder(recorder_dict, plot_realtime=plot_realtime, save=plot_save, save_path="rec")


"""Configure EIM"""

config = ConditionalMixtureEIM.get_default_config()
config.train_epochs = 1000
config.num_components = num_components

config.components_net_hidden_layers = [64, 64]
config.components_batch_size = 1000
config.components_num_epochs = 10
config.components_net_reg_loss_fact = 0.0
config.components_net_drop_prob = 0.0

config.gating_net_hidden_layers = [64, 64]
config.gating_batch_size = 1000
config.gating_num_epochs = 10
config.gating_net_reg_loss_fact = 0.0
config.gating_net_drop_prob = 0.0

config.dre_reg_loss_fact = 0.0005
config.dre_early_stopping = True
config.dre_drop_prob = 0.0
config.dre_num_iters = 50
config.dre_batch_size = 1000
config.dre_hidden_layers = [128, 128, 128]

"""Build and Run EIM"""
model = ConditionalMixtureEIM(config, train_samples=data.train_samples, seed=42 * 7, recorder=recorder, val_samples=data.val_samples)
model.train()

