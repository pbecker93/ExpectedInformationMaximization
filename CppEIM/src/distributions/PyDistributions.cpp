#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <PyArmaConverter.h>

#include <distributions/Categorical.h>
#include <distributions/Gaussian.h>
#include <distributions/GMM.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp_distributions, m) {
    /*** CATEGORICAL ***/
    py::class_<Categorical> cat(m, "Categorical");

    cat.def(py::init([](dpy_arr probabilities){
        return new Categorical(to_vec<double>(probabilities));}),
        py::arg("p"));

    cat.def("sample", [](Categorical *obj, uword size){return from_mat<uword>(obj->sample(size));},
            py::arg("size"));

    cat.def_property_readonly("entropy", &Categorical::entropy);

    cat.def("kl", &Categorical::kl, py::arg("other"));

    cat.def_property_readonly("probabilities",[](Categorical* obj){
        return from_mat<double>(obj->probabilities());});

    /*** GAUSSIAN ***/
    py::class_<Gaussian> gauss(m, "Gaussian");
    gauss.def(py::init([](dpy_arr mean, dpy_arr covar){
        return new Gaussian(to_vec<double>(mean), to_mat<double>(covar));}),
        py::arg("mean"), py::arg("covar"));

    gauss.def("density", [](Gaussian* obj, dpy_arr samples){
        return from_mat<double>(obj->density(to_mat<double>(samples)));},
        py::arg("samples"));

    gauss.def("log_density", [](Gaussian* obj, dpy_arr samples){
        return from_mat<double>(obj->log_density(to_mat<double>(samples)));},
        py::arg("samples"));

    gauss.def("log_likelihood", [](Gaussian *obj, dpy_arr samples){
        return obj->log_likelihood(to_mat<double>(samples));},
        py::arg("samples"));

    gauss.def("sample",[](Gaussian* obj, uword size){
        return from_mat<double>(obj->sample(size));},
        py::arg("num_samples"));

    gauss.def("kl", &Gaussian::kl, py::arg("other"));

    gauss.def_property_readonly("entropy", &Gaussian::entropy);

    gauss.def_property_readonly("mean", [](Gaussian* obj){return from_mat<double>(obj->get_mean());});

    gauss.def_property_readonly("covar", [](Gaussian *obj){return from_mat_enforce_mat<double>(obj->get_covar());});

    /*** GAUSSIAN MIXTURE ***/
    py::class_<GMM> gmm(m, "GMM");
    gmm.def(py::init([](dpy_arr weights, dpy_arr means, dpy_arr covars){
        return new GMM(to_vec<double>(weights), to_mat<double>(means), to_cube<double>(covars));}),
        py::arg("weights"), py::arg("means"), py::arg("covars"));

    gmm.def("density", [](GMM *obj, dpy_arr samples){
        return from_mat<double>(obj->density(to_mat<double>(samples)));}, py::arg("samples"));

    gmm.def("log_density", [](GMM *obj, dpy_arr samples){
        return from_mat<double>(obj->log_density(to_mat<double>(samples)));}, py::arg("samples"));

    gmm.def("log_likelihood", [](GMM *obj, dpy_arr samples){
        return obj->log_likelihood(to_mat<double>(samples));}, py::arg("samples"));

    gmm.def("sample", [](GMM *obj, uword size){return from_mat<double>(obj->sample(size));},
            py::arg("num_samples"));

    gmm.def("update_component_parameters", [](GMM *obj, uword index, dpy_arr new_mean, dpy_arr new_covar){
        obj->update_component_parameters(index, to_vec<double>(new_mean), to_mat<double>(new_covar));},
        py::arg("index"), py::arg("new_mean"), py::arg("new_covar"));

    gmm.def("update_weight_parameters", [](GMM *obj, dpy_arr new_p){
        obj->update_weight_parameters(to_vec<double>(new_p));}, py::arg("new_p"));

    gmm.def_property_readonly("components", &GMM::get_components);

    gmm.def_property_readonly("weight_distribution", &GMM::get_weight_distribution);

    gmm.def_property_readonly("num_components", &GMM::get_num_components);

}
