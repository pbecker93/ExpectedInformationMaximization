#ifndef GENERATOR_GMM_H
#define GENERATOR_GMM_H

#define ARMA_DONT_PRINT_ERRORS

#include <distributions/Gaussian.h>
#include <distributions/Categorical.h>

class GMM{


public:
    typedef std::vector<Gaussian> ComponentList;


    GMM();

    GMM(vec weights, mat means, cube covars);



    vec density(const mat &samples) const;
    vec log_density(const mat &samples) const ;

    double log_likelihood(const mat &samples) const;

    mat sample(uword size) const;

    uword get_num_components() const { return num_components;}

    ComponentList get_components() const {return components;}
    Categorical get_weight_distribution() const { return weight_distribution;}

    void update_weight_parameters(const vec& p) { weight_distribution.set_probabilities(p);}
    void update_component_parameters(uword i, const vec& mean, const mat& covar){
        components[i].update_parameters(mean, covar);}

private:
    ComponentList components;
    Categorical weight_distribution;
    uword dim;
    uword num_components;
};

#endif //GENERATOR_GMM_H
