#include <pybind11/pybind11.h>
#include <PyArmaConverter.h>
#include <regression/Regression.h>

namespace py = pybind11;

PYBIND11_MODULE(cpp_regression, m) {

    /*** Linear Regression ***/
    py::class_<LinFunc> lf(m, "LinFunc");

    lf.def(py::init([](uword input_dim, double reg_fact, bool normalize){
        return new LinFunc(input_dim, reg_fact, normalize);}),
        py::arg("input_dim"), py::arg("regularization_fact"), py::arg("normalize"));

    lf.def("__call__", [](LinFunc* obj, dpy_arr input){
        return from_mat<double>(obj->call(to_mat<double>(input)));},
        py::arg("x"));

    lf.def("fit", [](LinFunc* obj, dpy_arr input, dpy_arr output){
        obj->fit(to_mat<double>(input), to_vec<double>(output));},
        py::arg("x"), py::arg("y"));

    lf.def("fit_weighted", [](LinFunc* obj, dpy_arr input, dpy_arr output, dpy_arr weights){
        obj->fit(to_mat<double>(input), to_mat<double>(output), to_mat<double>(weights));},
        py::arg("x"), py::arg("y"), py::arg("weights"));

    lf.def_property_readonly("params", [](LinFunc* obj){return from_mat<double>(obj->get_params());});

    /*** QUAD FUNC ***/
    py::class_<QuadFunc> qf(m, "QuadFunc");

    qf.def(py::init([](uword input_dim, double reg_fact, bool normalize){
        return new QuadFunc(input_dim, reg_fact, normalize);}),
        py::arg("input_dim"), py::arg("regularization_fact"), py::arg("normalize"));

    qf.def("__call__", [](QuadFunc* obj, dpy_arr input){
        return from_mat<double>(obj->call(to_mat<double>(input)));},
        py::arg("x"));

    qf.def("fit", [](QuadFunc *obj, dpy_arr input, dpy_arr output){
        obj->fit(to_mat<double>(input), to_vec<double>(output));},
        py::arg("x"), py::arg("y"));

    qf.def("fit_whitened", [](QuadFunc *obj, dpy_arr input, dpy_arr output,
                              dpy_arr sample_mean, dpy_arr sample_covar_chol){
        obj->fit_whitened(to_mat<double>(input), to_vec<double>(output),
                          to_vec<double >(sample_mean), to_mat<double>(sample_covar_chol).t());},
        py::arg("x"), py::arg("y"), py::arg("x_mean"),
        py::arg("x_covar_chol"));

    qf.def_property_readonly("quad_term", [](QuadFunc *obj){
        return from_mat_enforce_mat<double>(obj->get_quad_term());});

    qf.def_property_readonly("lin_term", [](QuadFunc *obj){return from_mat<double>(obj->get_lin_term());});

    qf.def_property_readonly("const_term", &QuadFunc::get_const_term);

}