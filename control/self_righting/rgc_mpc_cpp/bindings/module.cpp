// Python bindings (pybind11) for the RGC-MPC C++ core.
// Only numpy arrays / Python scalars cross the boundary (no pinocchio / Boost.Python types).
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rgc_mpc/stage_qp.hpp"

namespace py = pybind11;
using rgc_mpc::Mat;
using rgc_mpc::Vec;

namespace {

std::vector<Mat> split_blocks(const Mat& stacked, int n_blocks, const char* name) {
    if (n_blocks <= 0 || stacked.rows() % n_blocks != 0)
        throw std::invalid_argument(std::string(name) + ": rows must be a multiple of N");
    const auto b = stacked.rows() / n_blocks;
    std::vector<Mat> out;
    out.reserve(n_blocks);
    for (int k = 0; k < n_blocks; ++k) out.emplace_back(stacked.middleRows(k * b, b));
    return out;
}

}  // namespace

PYBIND11_MODULE(_rgc_mpc_cpp, m) {
    m.doc() = "RGC-MPC C++ core: stage-wise MPC QP solved with HPIPM";

    py::class_<rgc_mpc::HpipmStageQpSolver>(m, "HpipmStageQpSolver")
        .def(py::init([](int N, int nx, int nu, int ng, std::vector<int> soft_rows, double tol, int iter_max,
                         std::string mode) {
                 rgc_mpc::SolverOptions o;
                 o.tol = tol;
                 o.iter_max = iter_max;
                 o.mode = std::move(mode);
                 return std::make_unique<rgc_mpc::HpipmStageQpSolver>(N, nx, nu, ng, std::move(soft_rows), o);
             }),
             py::arg("N"), py::arg("nx"), py::arg("nu"), py::arg("ng"), py::arg("soft_rows") = std::vector<int>{},
             py::arg("tol") = 1e-6, py::arg("iter_max") = 100, py::arg("mode") = "speed_abs")
        .def_property_readonly("N", &rgc_mpc::HpipmStageQpSolver::N)
        .def_property_readonly("nx", &rgc_mpc::HpipmStageQpSolver::nx)
        .def_property_readonly("nu", &rgc_mpc::HpipmStageQpSolver::nu)
        .def_property_readonly("ng", &rgc_mpc::HpipmStageQpSolver::ng)
        .def_property_readonly("ns", &rgc_mpc::HpipmStageQpSolver::ns)
        .def(
            "solve",
            [](rgc_mpc::HpipmStageQpSolver& self, const Mat& A, const Mat& B, const Mat& Cy, const Mat& Qy_stacked,
               const Mat& ref, const Mat& R_stacked, const Mat& C, const Mat& D, const Mat& lg, const Mat& ug,
               const Mat& zl, double z_quad, const Vec& x0) {
                rgc_mpc::StageQp qp;
                qp.A = A;
                qp.B = B;
                qp.Cy = Cy;
                qp.Qy = split_blocks(Qy_stacked, self.N(), "Qy");
                qp.ref = ref;
                qp.R = split_blocks(R_stacked, self.N(), "R");
                qp.C = C;
                qp.D = D;
                qp.lg = lg;
                qp.ug = ug;
                qp.zl = zl;
                qp.z_quad = z_quad;
                qp.x0 = x0;

                rgc_mpc::StageQpSolution s;
                {
                    py::gil_scoped_release release;  // heartbeat threads keep running during the solve
                    s = self.solve(qp);
                }

                py::dict out;
                out["status"] = s.status;
                out["iterations"] = s.iterations;
                out["res_stat"] = s.res_stat;
                out["res_eq"] = s.res_eq;
                out["res_ineq"] = s.res_ineq;
                out["res_comp"] = s.res_comp;
                out["U"] = s.U;
                out["X"] = s.X;
                out["S"] = s.S;
                out["lam_max"] = s.lam_max;
                out["tracking_cost"] = s.tracking_cost;
                out["effort_cost"] = s.effort_cost;
                out["slack_cost"] = s.slack_cost;
                out["solve_time"] = s.solve_time;
                return out;
            },
            py::arg("A"), py::arg("B"), py::arg("Cy"), py::arg("Qy"), py::arg("ref"), py::arg("R"), py::arg("C"),
            py::arg("D"), py::arg("lg"), py::arg("ug"), py::arg("zl"), py::arg("z_quad"), py::arg("x0"),
            R"doc(Solve the stage-wise QP.

Qy: (N*ny, ny) stacked output weights, R: (N*nu, nu) stacked input weights,
ref: (N, ny), lg/ug: (N, ng) with +-inf for unbounded sides, zl: (N, ns), x0: (nx,).
Returns a dict with status (0 = success), iterations, residuals, U (N, nu),
X (N+1, nx), S (N, ns), lam_max, tracking/effort/slack cost and solve_time [s].)doc");
}
