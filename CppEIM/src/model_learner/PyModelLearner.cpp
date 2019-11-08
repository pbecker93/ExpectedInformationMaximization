#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <PyArmaConverter.h>

#include <model_lerner/GMMLearner.h>
namespace py = pybind11;


PYBIND11_MODULE(cpp_model_learner, m) {
    /*** GMMLearner  ***/
    py::class_<GMMLearner> gl(m, "GMMLearner");

    gl.def(py::init([](uword dim, uword num_components, double surrogate_reg_fact, unsigned int seed,
                                double eta_offset, double omega_offset, bool constrain_entropy) {
        return new GMMLearner(dim, num_components, surrogate_reg_fact, seed, eta_offset, omega_offset,
                              constrain_entropy);}),
        py::arg("dim"), py::arg("num_components"), py::arg("surrogate_reg_fact"), py::arg("seed"),
        py::arg("eta_offset"), py::arg("omega_offset"), py::arg("constrain_entropy"));

    // Initialize Training
    gl.def("initialize_model", [](GMMLearner *obj, dpy_arr weights, dpy_arr means, dpy_arr covars) {
        obj->initialize_model(to_vec<double>(weights), to_mat<double>(means), to_cube<double>(covars));},
        py::arg("weights"), py::arg("means"), py::arg("covars"));

    // Update Components
    gl.def("update_components", [](
            GMMLearner *obj, std::vector<dpy_arr> samples, std::vector<dpy_arr>  rewards, double kl_bound,
            double entropy_loss_bound) {
        return obj->update_components(to_mats<double>(samples), to_vecs<double>(rewards),
                                      kl_bound, entropy_loss_bound);},
        py::arg("samples"), py::arg("rewards"), py::arg("kl_bound"), py::arg("entropy_loss_bound") = -1);

    // Update Weights
    gl.def("update_weights", [](
            GMMLearner *obj, dpy_arr rewards, double kl_bound, double entropy_loss_bound) {
        return obj->update_weights(to_vec<double>(rewards), kl_bound, entropy_loss_bound);},
        py::arg("rewards"), py::arg("kl_bound"), py::arg("entropy_loss_bound") = -1);

    gl.def_property_readonly("model", &GMMLearner::get_model);

}