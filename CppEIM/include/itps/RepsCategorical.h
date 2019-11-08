#ifndef CPPEIM_REPSCATEGORICAL_H
#define CPPEIM_REPSCATEGORICAL_H

#define ARMA_DONT_PRINT_ERRORS

#include <armadillo>
#include <distributions/Categorical.h>
#include <MathUtil.h>
#include <nlopt.hpp>
#include <NlOptUtil.h>


using namespace arma;

class RepsCategorical{

public:
    RepsCategorical(double eta_offset, double omega_offset, bool constrain_entropy);

    vec reps_step(double eps, double beta, const Categorical &old_dist, const vec &rewards);

    double get_last_eta() const { return eta;};
    double get_last_omega() const { return omega;};
    bool was_succ() const {return succ;}
    std::string get_res_txt() const { return res_txt;};

private:

    double dual(std::vector<double> const &eta_omega, std::vector<double> &grad);
    vec new_params(double eta, double omega);

    double eps, beta, eta_offset, omega_offset;
    bool constrain_entropy, succ;
    double eta=1, omega=1;
    std::vector<double> grad = std::vector<double>(2, 10);
    std::string res_txt;

    vec old_log_prob, rewards;
};

#endif //CPPEIM_REPSCATEGORICAL_H
