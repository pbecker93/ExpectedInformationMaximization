import numpy as np
from sklearn.cluster import k_means


def gmm_init(mode, samples, num_components, seed):
    rng = np.random.RandomState(seed)

    # random initialization
    if mode == "random":
        initial_weights = np.ones([num_components]) / num_components
        initial_means = samples[rng.choice(len(samples), num_components, replace=False)]
        initial_covars = np.tile(np.expand_dims(np.cov(samples.T), 0), [num_components, 1, 1])
        return initial_weights, initial_means, initial_covars

    # K means initialization
    elif mode == "kmeans":
        initial_responsibilities = np.zeros([len(samples), num_components])
        m, labels, _ = k_means(samples, num_components)
        initial_responsibilities[np.arange(len(samples)), labels] = 1.0
        cts = np.sum(initial_responsibilities, 0)
        initial_weights = cts / len(samples)
        initial_means = np.dot(initial_responsibilities.T, samples) / cts[:, np.newaxis]
        initial_covariances = np.zeros([num_components, samples.shape[1], samples.shape[1]])
        for i in range(num_components):
            initial_covariances[i] = np.cov(samples[labels == i], rowvar=False)
            initial_covariances[i] += 0.01 * np.eye(samples.shape[-1])
        return initial_weights, initial_means, initial_covariances

    # zero mean, unit covariance only for single gaussians
    elif mode == "zmuc":
        assert num_components == 1, "Zero Mean Unit Covariance Initialization currently only implemented for single component"
        dim = samples.shape[-1]
        return np.ones(1), np.zeros([1, dim]), np.expand_dims(np.eye(dim), 0)
    else:
        raise AssertionError("Invalid Initialization mode - either 'random' or 'kmeans' or 'zmuc' (i.e. zero mean unit covariance)")

