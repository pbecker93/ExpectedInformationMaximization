#ifndef GENERATOR_MATHUTIL_H
#define GENERATOR_MATHUTIL_H

#define ARMA_DONT_PRINT_ERRORS

#include <armadillo>

using namespace arma;

class MathUtil {

public:
    static double log_sum_exp(const vec &v) {
        double a = v.max();
        return a + log(sum(exp(v - a)) + 1e-150);
    }

    static rowvec log_sum_exp(const mat& m){
        rowvec a = max(m, 0);
        return a + log(sum(exp(m.each_row() - a), 0) + 1e-150);
    }

    static mat affine_mapping(const mat &x, const mat &w){
        mat linear = w.cols(0, w.n_cols - 2) * x;
        return linear.each_col() + w.col(w.n_cols - 1);
    }

    static mat softmax(const mat &x) {
        mat log_norm = log_sum_exp(x);
        return exp(x.each_row() - log_norm);
    }

    static mat get_rbf_features(const mat &x, const mat &centers, double sigma){
        mat features = mat(centers.n_cols, x.n_cols, fill::ones);
        for(uword i = 0; i < centers.n_cols; ++i){
            features.row(i) = exp(- (1 / sigma) * (sum(square(x.each_col() - centers.col(i)), 0)));
        }
        return features;
    }

};

#endif //GENERATOR_MATH_H
