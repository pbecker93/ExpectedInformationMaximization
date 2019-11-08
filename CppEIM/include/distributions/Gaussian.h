#ifndef GENERATOR_GAUSSIAN_H
#define GENERATOR_GAUSSIAN_H

#define ARMA_DONT_PRINT_ERRORS

#include <armadillo>
#include <MathUtil.h>

using namespace arma;

class Gaussian {

public:
    Gaussian(vec mean, mat covar);

    vec density (const mat &samples) const ;
    vec log_density (const mat &samples) const;
    double log_likelihood (const mat &samples) const;

    mat sample(uword size) const;

    double entropy() const;
    double kl(const Gaussian &other) const;
    double get_covar_logdet() const;

    //void update_natural_parameters(const vec& lin_term, const mat& quad_term);
    void update_parameters(const vec& mean, const mat& covar);

    vec get_mean() const { return mean_d;};
    mat get_covar() const { return covar;}
    vec get_lin() const  { return natural_lin;};
    mat get_quad() const { return natural_quad;};
    mat get_chol_covar() const {return chol_covar;};
    mat get_chol_quad_transposed() const { return chol_quad_transposed;};

private:
    vec exp_term(const mat &x) const;
   // vec exp_term2(const mat &x) const;
    void set_params(const vec& mean, const vec& lin_term,
                    const mat& covar, const mat& quad_term,
                    const mat& chol_covar, const mat& chol_quad_transposed);

    double norm_term;

    uword dim;

    vec mean_d, natural_lin;
    mat covar, natural_quad, chol_covar, chol_quad_transposed;
};

#endif //GENERATOR_GAUSSIAN_H
