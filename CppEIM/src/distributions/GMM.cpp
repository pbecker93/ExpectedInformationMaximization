#include <distributions/GMM.h>

GMM::GMM() :
    weight_distribution(vec(1, fill::ones)){
}

GMM::GMM(vec weights, mat means, cube covars) :
    weight_distribution(weights),
    dim(means.n_rows),
    num_components(weights.n_rows){
    mat covar_0 = covars.slice(0);
    for (uword i = 0; i < num_components; ++i){
        components.push_back(Gaussian(means.col(i), covars.slice(i)));
    }
}

vec GMM::density(const mat &samples) const {
    vec densities(samples.n_cols, fill::zeros);
    vec p = weight_distribution.probabilities();
    for (uword i = 0; i < num_components; ++i){
        densities += p.at(i) * components[i].density(samples);
    }
    return densities;
}

vec GMM::log_density(const mat &samples) const {
    return log(density(samples) + 1e-25);
}

double GMM::log_likelihood(const mat &samples) const{
    return mean(log_density(samples));
}

mat GMM::sample(uword size) const{
    uvec cts = weight_distribution.sample_cts(size);
    mat samples = mat(dim, size, fill::none);

    uvec end_idx = cumsum(cts);
    uvec start_idx = end_idx - cts;

    #pragma omp parallel for default(none) shared(cts, samples, start_idx, end_idx)
    for(uword i = 0; i < num_components; ++i){
        if (cts(i) > 0) {
            samples.cols(start_idx.at(i), end_idx.at(i) - 1) = components[i].sample(cts(i));
        }
    }
    uvec perm = shuffle(linspace<uvec>(0, size - 1, size));
    return samples.cols(perm);
}
