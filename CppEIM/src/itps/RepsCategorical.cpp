#include <itps/RepsCategorical.h>

RepsCategorical::RepsCategorical(double eta_offset, double omega_offset, bool constrain_entropy) :
    eta_offset(eta_offset), omega_offset(omega_offset), constrain_entropy(constrain_entropy)
{
}

vec RepsCategorical::reps_step(double eps, double beta, const Categorical &old_dist,const vec &rewards)
{
    this->eps = eps;
    this->beta = beta;
    succ = false;

    old_log_prob = old_dist.log_probabilities();
    this->rewards = rewards - mean(rewards);
    // opt
    nlopt::opt opt(nlopt::LD_LBFGS, 2);

    opt.set_min_objective([](const std::vector<double> &eta_omega, std::vector<double> &grad, void *instance){
        return ((RepsCategorical *) instance)->dual(eta_omega, grad);}, this);

    std::vector<double> opt_eta_omega;

    std::tie(succ, opt_eta_omega, res_txt) = NlOptUtil::opt_dual(opt);
    opt_eta_omega[1] = constrain_entropy ? omega : 0.0;

    if (!succ) {
        opt_eta_omega[0] = eta;
        opt_eta_omega[1] = constrain_entropy ? omega : 0.0;
        succ = NlOptUtil::valid_despite_failure(opt_eta_omega, grad);
    }

    vec res;
    if (succ) {
        eta = opt_eta_omega[0];
        omega = opt_eta_omega[1];
        res = new_params(eta, omega);
        res_txt = " ";
    } else{
        res_txt += "Failure, last grad " + std::to_string(grad[0]) + " " + std::to_string(grad[1]) + " - skipping ";
        res = vec();
    }
    return res;
}

vec RepsCategorical::new_params(double eta, double omega) {
    omega = constrain_entropy ? omega : 0.0;
    vec new_params = exp(((eta + eta_offset) * old_log_prob + rewards) / (eta + eta_offset + omega + omega_offset));
    return new_params / sum(new_params);
}

double RepsCategorical::dual(std::vector<double> const &eta_omega, std::vector<double> &grad){
    eta = eta_omega[0];
    omega = constrain_entropy ? eta_omega[1] : 0.0;

    if (eta < 0.0) {
        eta = 0.0;
    }
    if (omega < 0.0) {
        omega = 0.0;
    }
    double eta_off = eta + eta_offset;
    double omega_off = omega + omega_offset;

    vec t1 = (eta_off * old_log_prob + rewards) / (eta_off + omega_off);
    vec t1_de = (omega_off * old_log_prob - rewards) / (eta_off + omega_off); //one times (eta+omega) in denominator missing
    //t1_do = -t1  with one times (eta+omega) in denominator missing

    // log sum exp trick
    double t1_max = t1.max();
    vec exp_t1 = exp(t1 - t1_max);
    double sum_exp_t1 = sum(exp_t1) + 1e-25;
    double t2 = t1_max + log(sum_exp_t1);

    // factor of exp(t1_max) is still missing in sum_exp_t1
    double inv_sum = (1 / sum_exp_t1);
    // missing factors of exp(t1_max) in both inv_sum and exp_t1, cancel out here.
    double t2_de =   inv_sum * sum(t1_de % exp_t1);
    double t2_do = - inv_sum * sum(t1    % exp_t1); // -t2 =  t2_do

    grad[0] = eps + t2 + t2_de; // missing factor in t2_de cancels out with missing factor here
    grad[1] = constrain_entropy ? (- beta + t2 + t2_do) : 0.0; // missing factor in t2_do cancels out with missing factor here

    this->grad[0] = grad[0];
    this->grad[1] = grad[1];

    return eta * eps - omega * beta + (eta_off + omega_off) * t2;
}


