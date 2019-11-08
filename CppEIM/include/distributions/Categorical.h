#ifndef GENERATOR_CATEGORICAL_H
#define GENERATOR_CATEGORICAL_H

#define ARMA_DONT_PRINT_ERRORS

#include <armadillo>

using namespace arma;

class Categorical {
public:

    Categorical(vec probabilities);

    uvec sample(uword size) const;
    uvec sample_cts(uword size) const;


    vec probabilities() const {return probs;};
    vec log_probabilities() const {return log(probs + 1e-25);};

    void set_probabilities(const vec &new_probabilites){probs = new_probabilites;};

    double entropy() const;
    double kl(const Categorical &other) const;

private:

    vec probs;

};


#endif //GENERATOR_CATEGORICAL_H
