#include <model_lerner/GMMLearner.h>


GMMLearner::GMMLearner(uword dim, uword num_components, double surrogate_reg_fact, unsigned int seed,
                       double eta_offset, double omega_offset, bool constrain_entropy) :
        dim(dim),
        num_components(num_components),
        surrogate_reg_fact(surrogate_reg_fact),
        eta_offset(eta_offset), omega_offset(omega_offset), constrain_entropy(constrain_entropy),
        weight_learner(RepsCategorical(eta_offset, omega_offset, constrain_entropy))
{
    arma_rng::set_seed(seed);
    for(uword i = 0; i < num_components; ++i) {
        component_learners.emplace_back(MoreGaussian(dim, eta_offset, omega_offset, constrain_entropy));
    }
}

void GMMLearner::initialize_model(const vec &weights, const mat &means, const cube &covars) {
    model = GMM(weights, means, covars);
}

UpdateResVec
GMMLearner::update_components(const std::vector<mat> &samples, const std::vector<vec> &rewards,
                                double kl_bound, double entropy_loss_bound) {
    auto res = UpdateResVec(num_components);

    auto components = model.get_components();

    #pragma omp parallel for default(none) shared(samples, rewards, components, entropy_loss_bound, kl_bound, res)
    for(uword i = 0; i < num_components; ++i){
        Gaussian old_dist = Gaussian(components[i].get_mean(), components[i].get_covar());

        QuadFunc surrogate(dim, surrogate_reg_fact, true);
        surrogate.fit_whitened(samples[i], rewards[i], components[i].get_mean(), components[i].get_chol_covar());
        double entropy_bound = components[i].entropy() - entropy_loss_bound;
        vec new_mean;
        mat new_covar;
        std::tie(new_mean, new_covar) =
                component_learners[i].more_step(kl_bound, entropy_bound, components[i], surrogate);
        if (component_learners[i].was_succ()) {
            model.update_component_parameters(i, new_mean, new_covar);
        }

        double kl = model.get_components()[i].kl(old_dist);
        double entropy = model.get_components()[i].entropy();

        res[i] = std::make_tuple(kl, entropy, component_learners[i].get_last_eta(),
                 component_learners[i].get_last_omega(), component_learners[i].get_res_txt());
    }
    return res;
}

UpdateRes
GMMLearner::update_weights(const vec &rewards, double kl_bound, double entropy_loss_bound) {
    Categorical old_dist = Categorical(model.get_weight_distribution().probabilities());

    double entropy_bound = model.get_weight_distribution().entropy() - entropy_loss_bound;

    vec new_weights = weight_learner.reps_step(kl_bound, entropy_bound, old_dist, rewards);
    if (weight_learner.was_succ()){
        model.update_weight_parameters(new_weights);
    }

    double kl = model.get_weight_distribution().kl(old_dist);
    double entropy = model.get_weight_distribution().entropy();

    return std::make_tuple(kl, entropy, weight_learner.get_last_eta(), weight_learner.get_last_omega(),
                           weight_learner.get_res_txt());
}
