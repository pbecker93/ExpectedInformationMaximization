#include <regression/Regression.h>

inline void normalize_fn(mat &features, vec &outputs, vec &f_mean, vec &f_std,
                         double &o_mean, double &o_std, uword bias_entry){
    f_mean = mean(features, 1);
    f_std = stddev(features, 0, 1);
    if (bias_entry >= 0){
        f_mean[bias_entry] = 0.0;
        f_std[bias_entry] = 1.0;
    }
    features = (features.each_col() - f_mean).each_col() / f_std;
    o_mean = mean(outputs);
    o_std = stddev(outputs);
    outputs = (outputs - o_mean) / o_std;
}

inline void undo_normalization(vec &params, const vec &f_mean, const vec &f_std, const double o_mean,
                               const double o_std, uword bias_entry){
    params %= (o_std / f_std);
    params[bias_entry] = params[bias_entry] - dot(f_mean, params) + o_mean;
}

inline vec regress(const mat &weighted_features, const mat &features, const vec &outputs, double reg_fact){
    mat reg_mat = reg_fact * mat(features.n_rows, features.n_rows, fill::eye);
    vec params;
    bool succ = solve(params, weighted_features * features.t() + reg_mat, weighted_features * outputs);
    if (succ){
        return params;
    } else {
        throw std::logic_error("Inverting Matrix during fitting linear model failed!");
    }
}

inline vec linear_regression(mat &features, vec &outputs, double reg_fact, bool normalize, uword bias_entry){
    if(normalize) {
        vec f_mean, f_std;
        double o_mean, o_std;
        normalize_fn(features, outputs, f_mean, f_std, o_mean, o_std, bias_entry);
        vec params = regress(features, features, outputs, reg_fact);
        undo_normalization(params, f_mean, f_std, o_mean, o_std, bias_entry);
        return params;
    } else {
        return regress(features, features, outputs, reg_fact);
    }
}

inline vec weighted_linear_regression(mat &features, vec &outputs, const rowvec& weights, double reg_fact, bool normalize, uword bias_entry){
    if (normalize){
        vec f_mean, f_std;
        double o_mean, o_std;
        normalize_fn(features, outputs, f_mean, f_std, o_mean, o_std, bias_entry);
        mat weighted_features = features.each_row() % weights;
        vec params = regress(weighted_features, features, outputs, reg_fact);
        undo_normalization(params, f_mean, f_std, o_mean, o_std, bias_entry);
        return params;
    } else {
        mat weighted_features = features.each_row() % weights;
        return regress(weighted_features, features, outputs, reg_fact);
    }
}

/*** BASE CLASS REGRESSION FUNCTION ***/

RegressionFunc::RegressionFunc(double reg_fact, bool normalize) :
reg_fact(reg_fact),
normalize(normalize) {}

mat RegressionFunc::call(const mat &input) const{
    if(std::isnan(params.at(0, 0))){
        throw std::logic_error("Model needs to be fitted first");
    }
    mat features = feature_fn(input);
    return (params.t() * feature_fn(input));
}

void RegressionFunc::fit(const mat &inputs, const vec &outputs) {
    mat features = feature_fn(inputs);
    vec outputs_copy = outputs;
    params = linear_regression(features, outputs_copy, reg_fact, normalize, bias_entry());
}

void RegressionFunc::fit(const mat &inputs, const mat &outputs) {
    mat features = feature_fn(inputs);
    params = mat(features.n_rows, outputs.n_rows);
    for(uword i = 0; i < outputs.n_rows; ++i){
        vec outputs_copy = outputs.row(i).t();
        params.col(i) = linear_regression(features, outputs_copy, reg_fact, normalize, bias_entry());
    }
}


void RegressionFunc::fit(const mat &inputs, const vec &outputs, const rowvec &weights) {
    mat features = feature_fn(inputs);
    vec outputs_copy = outputs;
    params = weighted_linear_regression(features, outputs_copy, weights, reg_fact, normalize, bias_entry());
}

void RegressionFunc::fit(const mat &inputs, const mat &outputs, const mat &weights) {
    mat features = feature_fn(inputs);
    params = mat(features.n_rows, outputs.n_rows);
    for(uword i = 0; i < outputs.n_rows; ++i){
        vec outputs_copy = outputs.row(i).t();
        params.col(i) = weighted_linear_regression(features, outputs_copy, weights.row(i), reg_fact, normalize, bias_entry());
    }
}

/*** LINEAR REGRESSION ***/
mat LinFunc::feature_fn(const mat &raw) const{
    return join_cols(raw, mat(1, raw.n_cols, fill::ones));

}

/*** QUADRATIC REGRESSION ***/

QuadFunc::QuadFunc(uword input_dim, double reg_fact, bool normalize) :
    RegressionFunc(reg_fact, normalize),
    input_dim(input_dim)
{
    num_quad_features = static_cast<uword>(std::round((input_dim + 1) * (0.5 * input_dim)));
    feature_dim = num_quad_features + input_dim + 1;
    idx = find(trimatl(mat(input_dim, input_dim, fill::ones)));

    quad_term = mat(input_dim, input_dim, fill::none);
    lin_term = vec(input_dim, fill::none);
}

void QuadFunc::extract_terms() {
    vec params = get_params();
    quad_term.zeros();
    quad_term(idx) = params.rows(0, num_quad_features-1);
    quad_term = - quad_term - quad_term.t();

    lin_term = params.rows(num_quad_features, feature_dim-2);

    const_term = params.at(feature_dim - 1);
}

void QuadFunc::fit(const mat &inputs, const vec &outputs) {
    RegressionFunc::fit(inputs, outputs);
    extract_terms();
}

void QuadFunc::fit(const mat &inputs, const vec &outputs, const rowvec &weights) {
    RegressionFunc::fit(inputs, outputs, weights);
    extract_terms();
}
void QuadFunc::fit_whitened(const mat& inputs, const vec& outputs, const vec& sample_mean, const mat& sample_covar_chol) {
    mat inv_sample_covar_chol = inv(sample_covar_chol);
    mat white_inputs = whiten_inputs(inputs, sample_mean, inv_sample_covar_chol);
    RegressionFunc::fit(white_inputs, outputs);
    extract_terms(sample_mean, inv_sample_covar_chol);
}

void QuadFunc::fit_whitened(const mat& inputs, const vec& outputs, const rowvec &weights, const vec& sample_mean,
                            const mat& sample_covar_chol) {
    mat inv_sample_covar_chol = inv(sample_covar_chol);
    mat white_inputs = whiten_inputs(inputs, sample_mean, inv_sample_covar_chol);
    RegressionFunc::fit(white_inputs, outputs, weights);
    extract_terms(sample_mean, inv_sample_covar_chol);
}

mat QuadFunc::whiten_inputs(const mat& inputs, const vec& sample_mean, const mat& inv_sample_covar_chol) {
    return inv_sample_covar_chol * (inputs.each_col() - sample_mean);
}

void  QuadFunc::extract_terms(const vec& sample_mean, const mat& inv_sample_covar_chol) {
    vec params = get_params();
    quad_term.zeros();
    quad_term(idx) = params.rows(0, num_quad_features-1);
    quad_term = - quad_term - quad_term.t();
    quad_term = inv_sample_covar_chol.t() * quad_term * inv_sample_covar_chol;

    lin_term = params.rows(num_quad_features, feature_dim-2);
    vec b = inv_sample_covar_chol.t() * lin_term;
    lin_term = quad_term * sample_mean + b;
    const_term = params.at(feature_dim - 1) + dot(sample_mean, - 0.5 * quad_term * sample_mean - b);
}
