import numpy as np
from distributions import CppGMM
from distributions.GaussianEMM import GaussianEMM
import os


def save_model(model, path: str, filename: str):
    if isinstance(model, CppGMM):
        save_gmm(model, path, filename)
    elif isinstance(model, GaussianEMM):
        save_gaussian_gmm(model, path, filename)

    else:
        raise NotImplementedError("Saving not implemented for " + str(model.__class__))


def save_gaussian_gmm(model: GaussianEMM, path: str, filename: str):
    model.save(path, filename)


def save_gmm(model: CppGMM, path: str, filename: str):
    means = np.stack([c.mean for c in model.components], axis=0)
    covars = np.stack([c.covar for c in model.components], axis=0)
    model_dict = {"weights": model.weight_distribution.p, "means": means, "covars": covars}
    np.savez_compressed(os.path.join(path, filename + ".npz"), **model_dict)


def load_cpp_gmm(path: str, filename: str):
    model_path = os.path.join(path, filename + ".npz")
    model_dict = dict(np.load(model_path))
    return CppGMM(model_dict["weights"], model_dict["means"], model_dict["covars"])