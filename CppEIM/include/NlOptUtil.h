#ifndef GENERATOR_NLOPTUTIL_H
#define GENERATOR_NLOPTUTIL_H

#define ARMA_DONT_PRINT_ERRORS

#include <nlopt.hpp>
#include <tuple>

class NlOptUtil{
public:
    static std::string get_nlopt_err_string(nlopt::result opt_res){
        std::string err;
        switch (opt_res){
            case nlopt::FAILURE: err = "Failure"; break;
            case nlopt::INVALID_ARGS: err = "Invalid Arguments"; break;
            case nlopt::OUT_OF_MEMORY: err = "Out of Memory"; break;
            case nlopt::ROUNDOFF_LIMITED: err = "Roundoff Limited"; break;
            case nlopt::FORCED_STOP: err = "Force Stop"; break;
            case nlopt::SUCCESS: err = "Success"; break;
            case nlopt::STOPVAL_REACHED: err = "Stop Value Reached"; break;
            case nlopt::FTOL_REACHED: err = "FTOL Reached"; break;
            case nlopt::XTOL_REACHED: err = "XTOL Reached"; break;
            case nlopt::MAXEVAL_REACHED: err = "max eval reached"; break;
            case nlopt::MAXTIME_REACHED: err = "max time reached"; break;
            default: err = "None of above"; break;
            }
        return err;
    }

    static std::tuple<bool, std::vector<double>, std::string> opt_dual(nlopt::opt& opt){
        opt.set_lower_bounds(0);
        opt.set_upper_bounds(1e12);

        std::string add_txt;
        std::vector<double> eta_omega = std::vector<double>(2, 10.0);
        double dual_value;
        nlopt::result res;
        try{
            res = opt.optimize(eta_omega, dual_value);
        } catch (std::exception &ex){
            res = nlopt::FAILURE;
            add_txt = " " + static_cast<std::string>(ex.what());
        }
        return std::make_tuple(res > 0, eta_omega, get_nlopt_err_string(res) + add_txt);
    }

    static bool valid_despite_failure(std::vector<double>& eta_omega, std::vector<double>& grad){
        /*NLOPT sometimes throws errors because the dual and gradients do not fit together anymore for numerical reasons
         * This problem becomes severe for high dimensional data.
         * However, that happens mostly after the algorithm is almost converged. We check for those instances and just
         * work with the last values.
         *
         */
        //gradient norm close to 0
        double grad_bound = 1e-5;
        double value_bound = 1e-10;
        if (sqrt(grad[0] * grad[0] + grad[1] * grad[1]) < grad_bound){
            return true;
        }
        //omega at lower bound and gradient for eta close to 0
        if (eta_omega[1] < value_bound && grad[0] < grad_bound) {
            return true;
        }
        //eta at lower bound and gradient for omega close to 0
        if (eta_omega[0] < value_bound && grad[1] < grad_bound) {
            return true;
        }
        return false;
    }

};


#endif //GENERATOR_NLOPTUTIL_H
