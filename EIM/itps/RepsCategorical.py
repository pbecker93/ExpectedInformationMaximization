import numpy as np
from itps.ITPS import ITPS


class RepsCategorical(ITPS):

    def reps_step(self, eps, beta, old_dist, rewards):
        self._eps = eps
        self._beta = beta
        self._old_log_prob = old_dist.log_probabilities
        self._rewards = rewards

        try:
            opt_eta, opt_omega = self.opt_dual()
            new_params = self._new_params(opt_eta + self._eta_offset, opt_omega + self._omega_offset)
            self._succ = True
            return new_params
        except Exception:
            self._succ = False
            return None

    def _new_params(self, eta, omega):
        omega = omega if self._constrain_entropy else 0.0
        new_params = np.exp((eta * self._old_log_prob + self._rewards) / (eta + omega))
        return new_params / np.sum(new_params)

    def _dual(self, eta_omega, grad):
        eta = eta_omega[0] if eta_omega[0] > 0.0 else 0.0
        omega = eta_omega[1] if self._constrain_entropy and eta_omega[1] > 0.0 else 0.0
        self._eta = eta
        self._omega = omega

        eta_off = eta + self._eta_offset
        omega_off = omega + self._omega_offset

        t1 = (eta_off * self._old_log_prob + self._rewards) / (eta_off + omega_off)
        #  one times(eta + omega) in denominator  missing
        t1_de = (omega_off * self._old_log_prob - self._rewards) / (eta_off + omega_off)
        #  t1_do = -t1 with one times (eta+omega) in denominator missing

        t1_max = np.max(t1)
        exp_t1 = np.exp(t1 - t1_max)
        sum_exp_t1 = np.sum(exp_t1) + 1e-25
        t2 = t1_max + np.log(sum_exp_t1)

        #  factor of exp(t1_max) is still missing in sum_exp_t1
        inv_sum = (1 / sum_exp_t1)
        #  missing factors of exp(t1_max) in both inv_sum and exp_t1, cancel out here.
        t2_de =   inv_sum * np.sum(t1_de * exp_t1)
        t2_do = - inv_sum * np.sum(t1    * exp_t1)  #  -t2 =  t2_do

        grad[0] = self._eps + t2 + t2_de  # missing factor in t2_de cancels out with missing factor here
        #  missing factor in t2_do cancels out with missing factor here
        grad[1] = - self._beta + t2 + t2_do if self._constrain_entropy else 0.0

        self._grad[:] = grad

        dual = eta * self._eps - omega * self._beta + (eta_off + omega_off) * t2
        return dual
