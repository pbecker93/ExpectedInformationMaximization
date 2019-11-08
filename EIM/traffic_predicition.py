from recording.Recorder import Recorder
from recording.Recorder import RecorderKeys as rec_keys
from recording.modules.ModelModules import ModelRecMod
from recording.modules.DREModules import DRERecMod
from recording.modules.UpdateModules import WeightUpdateRecMod, ComponentUpdateRecMod
from recording.modules.SimpleModules import TrainIterationRecMod, ConfigInitialRecMod
import numpy as np
from data.TrafficData import StanfordData, LankershimData

from eim.MarginalMixtureEIM import MarginalMixtureEIM
import os
import matplotlib.pyplot as plt

"""Runs the traffic and pedestrian experiments described in section 5.3 of the paper"""

"""configure experiment"""
with_add_feat = True   # whether to use the additional features
dataset = "stanford"  # either 'stanford' or 'lankershim'


"""generate data"""
base_path = os.path.abspath(os.path.dirname(__file__))
if dataset.lower() == "stanford":
    path = os.path.join(base_path, "data/datasets/sdd")
    data = StanfordData(path, seed=42, trajectory_length=5)
elif dataset.lower() == "lankershim":
    path = os.path.join(base_path, "data/datasets/lankershim")
    data = LankershimData(path)
else:
    raise AssertionError("Invalid Dataset")

"""configure EIM"""
config = MarginalMixtureEIM.get_default_config()

config.num_components = 100
config.samples_per_component = 250
config.train_epochs = 1001
config.initialization = "kmeans"
# Component Updates
config.component_kl_bound = 0.05
# Mixture Updates
config.weight_kl_bound = 0.05
# Density Ratio Estimation
config.dre_reg_loss_fact = 0.0005  # Scaling Factor for L2 regularization of density ratio estimator
config.dre_early_stopping = True  # Use early stopping for density ratio estimator training
config.dre_drop_prob = 0.0  # If smaller than 1 dropout with keep prob = 'keep_prob' is used
config.dre_num_iters = 500  # Number of density ratio estimator steps each iteration (i.e. max number if early stopping)
config.dre_batch_size = 1000  # Batch size for density ratio estimator training
config.dre_hidden_layers = [256, 256, 256]  # width of density ratio estimator  hidden layers

"""Eval and plotting"""

img = data.get_overlay_img()


def eval_fn(model):
    model_samples = model.sample(data.test_samples.shape[0])

    plt.figure("test")
    plt.clf()
    if img is not None:
        plt.imshow(img)
    plt_samples = model.sample(20)
    plt_samples = data.de_standardize_data(np.reshape(plt_samples, [20, 5, 2]))
    for i in range(20):
        plt.scatter(plt_samples[i, :, 0], (plt_samples[i, :, 1]), c="orange")
        plt.plot(plt_samples[i, :, 0], (plt_samples[i, :, 1]), c="orange")
    plt.pause(0.001)
    return data.eval_fn(data.de_standardize_data(np.reshape(model_samples, [-1, 5, 2])).reshape([-1, 2]))


recorder_dict = {
    rec_keys.TRAIN_ITER: TrainIterationRecMod(),
    rec_keys.INITIAL: ConfigInitialRecMod(),
    rec_keys.MODEL: ModelRecMod(train_samples=data.train_samples,
                                eval_fn=eval_fn,
                                test_log_iters=10,
                                save_log_iters=10,
                                test_samples=data.test_samples),
    rec_keys.DRE: DRERecMod(data.train_samples),
    rec_keys.COMPONENT_UPDATE: ComponentUpdateRecMod(plot=False, summarize=True)
}
if config.num_components > 1:
    recorder_dict[rec_keys.WEIGHTS_UPDATE] = WeightUpdateRecMod(plot=True)

recorder = Recorder(recorder_dict, plot_realtime=True, save=False, save_path="rec")


""" run algorithm"""

eim = MarginalMixtureEIM(config, data.train_samples, recorder, val_samples=data.val_samples, seed=0,
                         add_feat_fn=data.get_add_feat if with_add_feat else None)
eim.train()



