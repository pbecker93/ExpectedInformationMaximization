import numpy as np
from distributions.marginal.Categorical import Categorical
from distributions.marginal.Gaussian import Gaussian


class GMM:

    def __init__(self, weights, means, covars):
        self._weight_distribution = Categorical(weights)
        self._components = [Gaussian(means[i], covars[i]) for i in range(means.shape[0])]

    def density(self, samples):
        densities = np.stack([self._components[i].density(samples) for i in range(self.num_components)], axis=0)
        w = np.expand_dims(self.weight_distribution.probabilities, axis=-1)
        return np.sum(w * densities, axis=0)

    def log_density(self, samples):
        return np.log(self.density(samples) + 1e-25)

    def log_likelihood(self, samples):
        return np.mean(self.log_density(samples))

    def sample(self, num_samples):
        w_samples = self._weight_distribution.sample(num_samples)
        samples =[]
        for i in range(self.num_components):
            cns = np.count_nonzero(w_samples == i)
            if cns > 0:
                samples.append(self.components[i].sample(cns))
        return np.random.permutation(np.concatenate(samples, axis=0))

    @property
    def components(self):
        return self._components

    @property
    def num_components(self):
        return len(self._components)

    @property
    def weight_distribution(self):
        return self._weight_distribution
