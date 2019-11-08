#include <itps/MoreGaussian.h>

MoreGaussian::MoreGaussian(uword dim, double eta_offset, double omega_offset, bool constrain_entropy) :
    dim(dim), eta_offset(eta_offset), omega_offset(omega_offset), constrain_entropy(constrain_entropy)
{
    dual_const_part = dim * log(2 * M_PI);
    entropy_const_part = 0.5 * (dual_const_part + dim);
}

std::tuple<vec, mat> MoreGaussian::more_step(double eps, double beta, const Gaussian &old_dist, const QuadFunc &reward)
{
    this->eps = eps;
    this->beta = beta;
    succ = false;
    eta_inc_ct = 0;

    old_mean = old_dist.get_mean();
    old_lin = old_dist.get_lin();
    old_chol_quad_transposed = old_dist.get_chol_quad_transposed();
    old_quad = old_dist.get_quad();

    reward_lin = reward.get_lin_term();
    reward_quad = reward.get_quad_term();


    // - logdet(Q_old)
    double old_logdet = old_dist.get_covar_logdet();

    // - 0.5 q^T_old Q_old^-1 q_old + 0.5 * logdet(Q_old) -0.5 * dim * log(2pi)
    old_term = -0.5 * (dot(old_lin, old_mean) + old_logdet + dual_const_part);

    kl_const_part = old_logdet - dim;

    // opt
    nlopt::opt opt(nlopt::LD_LBFGS, 2);

    opt.set_min_objective([](const std::vector<double> &eta_omega, std::vector<double> &grad, void *instance){
        return ((MoreGaussian *) instance)->dual(eta_omega, grad);}, this);

    std::vector<double> opt_eta_omega;

    std::tie(succ, opt_eta_omega, res_txt) = NlOptUtil::opt_dual(opt);
    opt_eta_omega[1] = constrain_entropy ? omega : 0.0;

    if (!succ) {
        opt_eta_omega[0] = eta;
        opt_eta_omega[1] = constrain_entropy ? omega : 0.0;
        succ = NlOptUtil::valid_despite_failure(opt_eta_omega, grad);
    }

    std::tuple<vec, mat> res;
    if (succ) {
        eta = opt_eta_omega[0];
        omega = opt_eta_omega[1];

        vec new_lin;
        mat new_covar;
        std::tie(new_lin, new_covar) = new_params_internal(eta, omega);
        res_txt = " ";
        res = std::make_tuple(new_covar * new_lin, new_covar);
    } else{
        res_txt += "Failure, last grad " + std::to_string(grad[0]) + " " + std::to_string(grad[1]) + " - skipping ";
        res = std::make_tuple(vec(), mat());
    }
    if (eta_inc_ct > 0){
        res_txt += "Increased eta " + std::to_string(eta_inc_ct) + " times.";
    }
    return res;
}

std::tuple<vec, mat> MoreGaussian::new_params_internal(double eta, double omega) {
    omega = constrain_entropy ? omega : 0.0;
    vec new_lin = ((eta + eta_offset) * old_lin + reward_lin) / (eta + eta_offset + omega + omega_offset);
    mat new_covar;
    while (eta < 1e12) {
        try {
            new_covar = inv_sympd((eta + eta_offset) * old_quad + reward_quad) * (eta + eta_offset + omega + omega_offset);
            break;
        } catch (std::runtime_error &err) {
            if (eta < 1e-12){
                eta = 1e-12;
            }
            eta *= 1.1;
            eta_inc_ct++;
        }
    }
    return std::make_tuple(new_lin, new_covar);

}

double MoreGaussian::dual(std::vector<double> const &eta_omega, std::vector<double> &grad){
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

    vec new_lin;
    mat new_covar;
    std::tie(new_lin, new_covar) = new_params_internal(eta, omega);
    mat new_chol_covar = chol(new_covar, "lower");

    vec new_mean = new_covar * new_lin;
    double new_logdet = 2 * sum(log(diagvec(new_chol_covar) + 1e-25));

    double dual = eta * eps - omega * beta + eta_off * old_term;
    dual += 0.5 * (eta_off + omega_off) * (dual_const_part + new_logdet + dot(new_lin, new_mean));

    vec diff = old_mean - new_mean;
    double trace_term = accu(square(old_chol_quad_transposed * new_chol_covar));
    double kl = 0.5 * (sum(square(old_chol_quad_transposed * diff)) + kl_const_part - new_logdet + trace_term);

    grad[0] = eps - kl;
    grad[1] = constrain_entropy ? (entropy_const_part + 0.5 * new_logdet - beta) : 0.0;
    this->grad[0] = grad[0];
    this->grad[1] = grad[1];
    return dual;
}
