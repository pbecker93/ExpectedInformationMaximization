import numpy as np
import nlopt

class ITPS:

    grad_bound = 1e-5
    value_bound = 1e-10

    def __init__(self, eta_offset, omega_offset, constrain_entropy):
        self._constrain_entropy = constrain_entropy
        self._eta_offset = eta_offset
        self._omega_offset = omega_offset

        self._eta = None
        self._omega = None
        self._grad = np.zeros(2)
        self._succ = False

    def opt_dual(self):
        opt = nlopt.opt(nlopt.LD_LBFGS, 2)
        opt.set_lower_bounds(0.0)
        opt.set_upper_bounds(1e12)
        opt.set_min_objective(self._dual)
        try:
            opt_eta_omega = opt.optimize([10.0, 10.0])
            opt_eta = opt_eta_omega[0]
            opt_omega = opt_eta_omega[1] if self._constrain_entropy else 0.0
            return opt_eta, opt_omega
        except Exception as e:
            # NLOPT somtimes throws error very close to convergence, we check for this and return preliminary result
            # if its close enough:
            # 1.Case: Gradient near 0
            # 2.Case: eta near bound and d_omega near 0
            # 3.Case: omega near bound and d_eta near 0
            if (np.sqrt(self._grad[0]**2 + self._grad[1]**2) < ITPS.grad_bound) or \
               (self._eta < ITPS.value_bound and np.abs(self._grad[1]) < ITPS.grad_bound) or \
               (self._omega < ITPS.value_bound and np.abs(self._grad[0]) < ITPS.grad_bound):
                return self._eta, self._omega
            else:
                raise e

    def _dual(self, eta_omega, grad):
        raise NotImplementedError

    @property
    def last_eta(self):
        return self._eta

    @property
    def last_omega(self):
        return self._omega

    @property
    def last_grad(self):
        return self._grad

    @property
    def success(self):
        return self._succ

    @property
    def eta_offset(self):
        return self._eta_offset

    @eta_offset.setter
    def eta_offset(self, new_eta_offset):
        self._eta_offset = new_eta_offset

    @property
    def omega_offset(self):
        return self._omega_offset

    @omega_offset.setter
    def omega_offset(self, new_omega_offset):
        self._omega_offset = new_omega_offset
