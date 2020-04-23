import numpy as np


class RegressionFunc:

    def __init__(self, reg_fact, normalize, unnormalize_output, bias_entry=None):
        self._reg_fact = reg_fact
        self._normalize = normalize
        self._unnormalize_output = unnormalize_output
        self._bias_entry = bias_entry
        self._params = None
        self.o_std = None

    def __call__(self, inputs):
        if self._params is None:
            raise AssertionError("Model not trained yet")
        return self._feature_fn(inputs) @ self._params

    def _feature_fn(self, x):
        raise NotImplementedError

    def _normalize_features(self, features):
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        # do not normalize bias
        if self._bias_entry is not None:
            mean[self._bias_entry] = 0.0
            std[self._bias_entry] = 1.0
        features = (features - mean) / std
        return features, np.squeeze(mean, axis=0), np.squeeze(std, axis=0)

    def _normalize_outputs(self, outputs):
        mean = np.mean(outputs)
        std = np.std(outputs)
        outputs = (outputs - mean) / std
        return outputs, mean, std

    def _undo_normalization(self, params, f_mean, f_std, o_mean, o_std):
        if self._unnormalize_output:
            params *= (o_std / f_std)
            params[self._bias_entry] = params[self._bias_entry] - np.dot(params, f_mean) + o_mean
        else:
            params *= (1.0 / f_std)
            params[self._bias_entry] = params[self._bias_entry] - np.dot(params, f_mean)
        return params

    def fit(self, inputs, outputs, weights=None):
        if len(outputs.shape) > 1:
            outputs = np.squeeze(outputs)
        features = self._feature_fn(inputs)
        if self._normalize:
            features, f_mean, f_std = self._normalize_features(features)
            outputs, o_mean, o_std = self._normalize_outputs(outputs)

        if weights is not None:
            if len(weights.shape) == 1:
                weights = np.expand_dims(weights, 1)
            weighted_features = weights * features
        else:
            weighted_features = features

        #regression
        reg_mat = np.eye(weighted_features.shape[-1]) * self._reg_fact
        if self._bias_entry is not None:
            reg_mat[self._bias_entry, self._bias_entry] = 0.0
        try:
            self._params = np.linalg.solve(weighted_features.T @ features + reg_mat, weighted_features.T @ outputs)
            if self._normalize:
                self._undo_normalization(self._params, f_mean, f_std, o_mean, o_std)
                self.o_std = o_std
        except np.linalg.LinAlgError as e:
            print("Error during matrix inversion", e.what())


class LinFunc(RegressionFunc):

    def __init__(self, reg_fact, normalize, unnormalize_output):
        super().__init__(reg_fact, normalize, unnormalize_output, -1)

    def _feature_fn(self, x):
        return np.concatenate([x, np.ones([x.shape[0], 1], dtype=x.dtype)], 1)


class QuadFunc(RegressionFunc):
    #*Fits - 0.5 * x ^ T  Rx + x ^ T r + r_0 ** * /

    def __init__(self, reg_fact, normalize, unnormalize_output):
        super().__init__(reg_fact, normalize, unnormalize_output, bias_entry=-1)
        self.quad_term = None
        self.lin_term = None
        self.const_term = None
        self._square_feat_upper_tri_ind = None

    def __square_feat_upper_tri_ind(self, n):
        if self._square_feat_upper_tri_ind is None or self._square_feat_upper_tri_ind.shape[0] != n:
            self._square_feat_upper_tri_ind = np.triu_indices(n)
        return self._square_feat_upper_tri_ind

    def _feature_fn(self, x):
        lin_feat = x
        quad_feat = np.transpose((x[:, :, np.newaxis] @ x[:, np.newaxis, :]), [1, 2, 0])
        quad_feat = quad_feat[self.__square_feat_upper_tri_ind(x.shape[-1])].T
        features = np.hstack([quad_feat, lin_feat, np.ones([x.shape[0], 1])])
        return features

    def fit(self, inputs, outputs, weights=None, sample_mean=None, sample_chol_cov=None):
        if sample_mean is None:
            assert sample_chol_cov is None
        if sample_chol_cov is None:
            assert sample_mean is None

        #whithening
        if sample_mean is not None and sample_chol_cov is not None:
            inv_samples_chol_cov = np.linalg.inv(sample_chol_cov)
            inputs = (inputs - sample_mean) @ inv_samples_chol_cov.T

        dim = inputs.shape[-1]

        super().fit(inputs, outputs, weights)

        idx = np.triu(np.ones([dim, dim], np.bool))

        qt = np.zeros([dim, dim])
        qt[idx] = self._params[:- (dim + 1)]
        self.quad_term = - qt - qt.T

        self.lin_term = self._params[-(dim + 1): -1]
        self.const_term = self._params[-1]

        #unwhitening:
        if sample_mean is not None and sample_chol_cov is not None:
            self.quad_term = inv_samples_chol_cov.T @ self.quad_term @ inv_samples_chol_cov
            t1 = inv_samples_chol_cov.T @ self.lin_term
            t2 = self.quad_term @ sample_mean
            self.lin_term = t1 + t2
            self.const_term += np.dot(sample_mean, -0.5 * t2 - t1)
