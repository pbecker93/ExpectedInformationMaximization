#ifndef GENERATOR_GMMGENERATOR_H
#define GENERATOR_GMMGENERATOR_H

#define ARMA_DONT_PRINT_ERRORS

#include <armadillo>
#include <regression/Regression.h>
#include <itps/RepsCategorical.h>
#include <itps/MoreGaussian.h>
#include <distributions/GMM.h>
#include <distributions/Gaussian.h>
#include <MathUtil.h>

using namespace arma;

typedef std::tuple<double, double, double, double, std::string> UpdateRes;
typedef std::vector<UpdateRes> UpdateResVec;

class GMMLearner{

 public:
    GMMLearner(uword dim, uword num_components, double surrogate_reg_fact, unsigned int seed,
               double eta_offset, double omega_offset, bool constrain_entropy);

    void initialize_model(const vec &weights, const mat &means, const cube &covars);

    UpdateResVec update_components(const std::vector<mat> &samples, const std::vector<vec> &rewards,
                                   double kl_bound, double entropy_loss_bound);

    UpdateRes update_weights(const vec& rewards, double kl_bound, double entropy_loss_bound);

    GMM get_model() const {return model;};

private:

    uword dim, num_components;
    double surrogate_reg_fact, eta_offset, omega_offset;
    GMM model;
    bool constrain_entropy;

    RepsCategorical weight_learner;
    std::vector<MoreGaussian> component_learners;

};


#endif //GENERATOR_GMMGENERATOR_H
