#ifndef GENERATOR_REGRESSION_H
#define GENERATOR_REGRESSION_H

#define ARMA_DONT_PRINT_ERRORS

#include <armadillo>
#include <MathUtil.h>

using namespace arma;

class RegressionFunc {

public:
    RegressionFunc(double reg_fact, bool normalize);


    mat call(const mat& input) const;

    void fit(const mat& inputs, const vec& outputs, const rowvec& weights);

    void fit(const mat& inputs, const vec& outputs);

    void fit(const mat& inputs, const mat& outputs, const mat& weights);

    void fit(const mat& inputs, const mat& outputs);


    mat get_params() const {return params;};

private:
    mat params = std::nan("") * mat(1,1, fill::ones);
    double reg_fact;
    bool normalize;
    virtual mat feature_fn(const mat& raw) const = 0;
    virtual uword bias_entry() const = 0;


};

class LinFunc : public RegressionFunc {
public:
    LinFunc(uword input_dim, double reg_fact, bool normalize) :
        RegressionFunc(reg_fact, normalize), input_dim(input_dim) {};
private:
    uword bias_entry() const override { return input_dim;};

    mat feature_fn(const mat &raw) const override;
    uword input_dim;
};

static mat get_quad_features(const mat &x, bool with_bias) {
    auto num_quad_features = static_cast<uword>(std::round((x.n_rows + 1) * (0.5 * x.n_rows)));
    uword feature_dim = num_quad_features + x.n_rows + (with_bias ? 1 : 0);
    mat features = mat(feature_dim, x.n_cols, fill::none);
    for(uword i = 0; i < x.n_cols; ++i){
        uword write_idx = 0;
        for(uword j = 0; j < x.n_rows; ++j){
            for(uword k = 0; k < (x.n_rows - j); ++k){
                //at faster than () since no bound check performed
                features.at(write_idx, i) = x.at(j, i) * x.at(j+k, i);
                write_idx++;
            }
        }
        features.submat(num_quad_features, i, num_quad_features + x.n_rows - 1, i) = x.col(i);
        if (with_bias){
            features.at(feature_dim - 1, i) = 1.0;
        }
    }
    return features;
}

class QuadFunc : public RegressionFunc {
/***Fits -0.5 * x^T Rx + x^T r + r_0 ***/
public:
    QuadFunc(uword input_dim, double reg_fact, bool normalize);

    void fit(const mat& inputs, const vec& outputs);
    void fit(const mat& inputs, const vec& outputs, const rowvec& weights);
    void fit_whitened(const mat& inputs, const vec& outputs, const vec& sample_mean, const mat& sample_covar_chol);
    void fit_whitened(const mat& inputs, const vec& outputs, const rowvec& weights, const vec& sample_mean,
                      const mat& sample_covar_chol);

    mat get_quad_term() const {return quad_term;};
    vec get_lin_term() const {return lin_term;};
    double get_const_term() const{return const_term;};

private:
    uword bias_entry() const override {return feature_dim - 1;};

    mat whiten_inputs(const mat& inputs, const vec& sample_mean, const mat& inv_sample_covar_chol);
    mat feature_fn (const mat &raw) const override {return get_quad_features(raw, true);};

    void extract_terms();
    void extract_terms(const  vec& sample_mean, const mat& inv_sample_covar_chol);


    uword input_dim, feature_dim, num_quad_features;

    uvec idx;
    mat quad_term;
    vec lin_term;
    double const_term;

};

#endif //GENERATOR_REGRESSION_H
