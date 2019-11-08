#ifndef GENERATOR_GAUSSIANMOREDUAL_H
#define GENERATOR_GAUSSIANMOREDUAL_H

#define ARMA_DONT_PRINT_ERRORS

#include <armadillo>
#include <distributions/Gaussian.h>
#include <regression/Regression.h>
#include <MathUtil.h>
#include <nlopt.hpp>
#include <NlOptUtil.h>


using namespace arma;

class MoreGaussian{

public:
    MoreGaussian(uword dim, double eta_offset, double omega_offset, bool constrain_entropy);

    std::tuple<vec, mat> more_step(double eps, double beta, const Gaussian &old_dist, const QuadFunc &reward);

    double get_last_eta() const { return eta;};
    double get_last_omega() const { return omega;};
    bool was_succ() const {return succ;}
    std::string get_res_txt() const { return res_txt;};

private:

    double dual(std::vector<double> const &eta_omega, std::vector<double> &grad);
    std::tuple<vec, mat> new_params_internal(double eta, double omega);

    double eps, beta, eta_offset, omega_offset;
    bool constrain_entropy, succ;
    uword dim, eta_inc_ct;
    double eta=1, omega=1;
    std::vector<double> grad = std::vector<double>(2, 10);

    double dual_const_part, old_term, entropy_const_part, kl_const_part;

    std::string res_txt;

    vec old_lin, old_mean, reward_lin;
    mat old_quad, old_chol_quad_transposed, reward_quad;

};

#endif //GENERATOR_GAUSSIANMOREDUAL_H
