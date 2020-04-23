import numpy as np
from itps.ITPS import ITPS


class MoreGaussian(ITPS):

    def __init__(self, dim, eta_offset, omega_offset, constrain_entropy):
        super().__init__(eta_offset, omega_offset, constrain_entropy)

        self._dim = dim
        self._dual_const_part = dim * np.log(2 * np.pi)
        self._entropy_const_part = 0.5 * (self._dual_const_part + dim)

    def more_step(self, eps, beta, old_dist, reward_surrogate):
        self._eps = eps
        self._beta = beta
        self._succ = False

        self._old_lin_term = old_dist.lin_term
        self._old_precision = old_dist.precision
        self._old_mean = old_dist.mean
        self._old_chol_precision_t = old_dist.chol_precision.T

        self._reward_lin_term = reward_surrogate.lin_term
        self._reward_quad_term = reward_surrogate.quad_term

        old_logdet = old_dist.covar_logdet()
        self._old_term = -0.5 * (np.dot(self._old_lin_term, self._old_mean) + self._dual_const_part + old_logdet)
        self._kl_const_part = old_logdet - self._dim

        try:
            opt_eta, opt_omega = self.opt_dual()
            new_lin, new_precision = self._new_params(opt_eta + self._eta_offset, opt_omega + self._omega_offset)
            new_covar = np.linalg.inv(new_precision)
            new_mean = new_covar @ new_lin
            self._succ = True
            return new_mean, new_covar
        except Exception:
            self._succ = False
            return None, None

    def _new_params(self, eta, omega):
        new_lin = (eta * self._old_lin_term + self._reward_lin_term) / (eta + omega)
        new_precision = (eta * self._old_precision + self._reward_quad_term) / (eta + omega)
        return new_lin, new_precision

    def _dual(self, eta_omega, grad):
        eta = eta_omega[0] if eta_omega[0] > 0.0 else 0.0
        omega = eta_omega[1] if self._constrain_entropy and eta_omega[1] > 0.0 else 0.0
        self._eta = eta
        self._omega = omega

        eta_off = eta + self._eta_offset
        omega_off = omega + self._omega_offset

        new_lin, new_precision = self._new_params(eta_off, omega_off)
        try:
            new_covar = np.linalg.inv(new_precision)
            new_chol_covar = np.linalg.cholesky(new_covar)

            new_mean = new_covar @ new_lin
            new_logdet = 2 * np.sum(np.log(np.diagonal(new_chol_covar) + 1e-25))

            dual = eta * self._eps - omega * self._beta + eta_off * self._old_term
            dual += 0.5 * (eta_off + omega_off) * (self._dual_const_part + new_logdet + np.dot(new_lin, new_mean))

            trace_term = np.sum(np.square(self._old_chol_precision_t @ new_chol_covar))
            kl = self._kl_const_part - new_logdet + trace_term
            diff = self._old_mean - new_mean
            kl = 0.5 * (kl + np.sum(np.square(self._old_chol_precision_t @ diff)))

            grad[0] = self._eps - kl
            grad[1] = (self._entropy_const_part + 0.5 * new_logdet - self._beta) if self._constrain_entropy else 0.0
            self._grad[:] = grad
            return dual

        except np.linalg.LinAlgError as e:
            grad[0] = -1.0
            grad[1] = 0.0
            return 1e12