#include <distributions/Categorical.h>

Categorical::Categorical(vec probabilities) :
probs(probabilities){
    double s = sum(probs);
    if ( s < 1 - 1e-12 or s > 1 + 1e-12){
        throw std::invalid_argument("Sum of probabilities " + std::to_string(s) + ", needs to be 1.0");
    }
}

uvec Categorical::sample(uword size) const{
    uword i, j;
    vec borders = cumsum(probs);
    borders.at(probs.size() - 1) = 1.0;
    vec raw_samples = randu(size);
    uvec samples = uvec(size, fill::none);
    for(i = 0; i < size; ++i) {
        for (j = 0; j < probs.size(); ++j) {
            if (raw_samples.at(i) <= borders.at(j))
                break;
        }
        samples.at(i) = j;
    }
    return samples;
}

uvec Categorical::sample_cts(uword size) const{
    uword i, j;
    vec borders = cumsum(probs);
    vec raw_samples = randu(size);
    uvec cts = uvec(probs.size(), fill::zeros);
    for(i = 0; i < size; ++i) {
        for (j = 0; j < probs.size(); ++j) {
            if (raw_samples.at(i) <= borders.at(j))
                break;
        }
        cts.at(j) += 1;
    }
    return cts;
}

double Categorical::entropy() const{
    return - sum(probs % log(probs + 1e-25));
}

double Categorical::kl(const Categorical &other) const {
    return sum(probs % (log(probs + 1e-25) - log(other.probabilities() + 1e-25)));
}