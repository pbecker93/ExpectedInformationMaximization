import numpy as np
from distributions import CppGMM

class GMMData:

    def __init__(self, weights, means, covars, num_train_samples, num_test_samples, seed,
                 num_val_samples=0):

        self._rng = np.random.RandomState(seed)

        self._dim = means.shape[-1]
        self.weights = weights
        self.means = means
        self.covars = covars

        self.model = CppGMM(self.weights, self.means, self.covars)

        self.train_samples = self.sample(num_train_samples)
        self.test_samples = self.sample(num_test_samples)
        if num_val_samples > 0:
            self.val_samples = self.sample(num_val_samples)

    def sample(self, num_samples):
        modes = self._rng.choice(a=len(self.weights), size=num_samples, p=self.weights)
        cts = [np.count_nonzero(modes == i) for i in range(len(self.weights))]
        samples = [self._rng.multivariate_normal(self.means[i], self.covars[i], cts[i]) for i in range(len(self.weights))]
        return self._rng.permutation(np.concatenate(samples, 0))

    @property
    def dim(self):
        return self._dim

    def log_density(self, samples):
        return self.model.log_density(samples)


class RandomGMMData(GMMData):

    VIPS_PARAMS = [20, 10, [-50, 50], 20]

    def __init__(self, dimensionality, num_components, mean_limits, covar_variance,
                 num_train_samples, num_test_samples, seed=0, num_val_samples=0):
        self._rng = np.random.RandomState(seed=seed)
        weights = self._rng.rand(num_components)
        weights /= np.sum(weights)

        means = self._rng.uniform(low=mean_limits[0], high=mean_limits[1], size=[num_components, dimensionality])

        covars = self._rng.normal(0, np.sqrt(covar_variance), size=[num_components, dimensionality, dimensionality])
        covars = np.matmul(np.transpose(covars, [0, 2, 1]), covars) + np.expand_dims(np.eye(dimensionality), 0) / 50

        super().__init__(weights, means, covars, num_train_samples, num_test_samples, seed,
                         num_val_samples)

