#include <distributions/Gaussian.h>
#include <MathUtil.h>

Gaussian::Gaussian(vec mean, mat covar){
    dim = mean.n_rows;
    update_parameters(mean, covar);
}

mat Gaussian::sample(uword size) const {
    mat t = chol_covar * randn(dim, size);
    mat res = t.each_col() + mean_d;
    return res;
}

vec Gaussian::density(const mat &samples) const {
    return norm_term * exp(-0.5 * exp_term(samples));
}

vec Gaussian::log_density(const mat &samples) const {
    return -0.5 * (dim * log(2 * M_PI) + get_covar_logdet() + exp_term(samples));
}

double Gaussian::log_likelihood(const mat &samples) const {
    return mean(log_density(samples));
}

double Gaussian::entropy() const {
    return 0.5 * (dim * log(2 * M_PI * M_E) + get_covar_logdet());
}

double Gaussian::kl(const Gaussian &other) const {
    mat other_chol_quad_transposed = other.get_chol_quad_transposed();
    double trace_term = accu(square(other_chol_quad_transposed * chol_covar));
    double kl = other.get_covar_logdet() - get_covar_logdet() - dim + trace_term;
    vec diff = other.get_mean() - mean_d;
    kl = kl + sum(square(other_chol_quad_transposed * diff));
    return 0.5 * kl;

}

vec Gaussian::exp_term(const mat &x) const {
    mat diff = x.each_col() - mean_d;
    return sum(square(chol_quad_transposed * diff), 0).t();
}

void Gaussian::update_parameters(const vec& mean, const mat& covar) {
    try {
        mat chol_cov = trimatl(chol(covar, "lower"));
        mat inv_chol_cov = inv(chol_cov);
        mat quad = inv_chol_cov.t() * inv_chol_cov;
        mat chol_quad_transposed = trimatl(chol(quad, "lower")).t();
        vec lin = quad * mean;

        set_params(mean, lin, covar, quad, chol_cov, chol_quad_transposed);
    } catch (std::runtime_error &err) {
        std::cout << "PARAMETER UPDATE REJECTED: " << err.what() << std::endl;
    }
}

void Gaussian::set_params(const vec& mean, const vec& lin_term, const mat& covar, const mat& quad_term,
                          const mat& chol_covar, const mat& chol_quad_transposed) {
    this->mean_d =mean;
    this->covar = covar;

    this->natural_lin = lin_term;
    this->natural_quad = quad_term;

    this->chol_covar = chol_covar;
    this->chol_quad_transposed = chol_quad_transposed;

    norm_term = 1 / sqrt(pow((2 * M_PI), dim) * exp(get_covar_logdet()));
}

double Gaussian::get_covar_logdet() const {
    return 2 * sum(log(diagvec(chol_covar) + 1e-25));
}

