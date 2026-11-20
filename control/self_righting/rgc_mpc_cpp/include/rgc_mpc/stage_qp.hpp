// Stage-wise linear MPC QP used by every RGC mode, and its HPIPM solver.
//
// Augmented state s = [x; u_prev] (dim nx), input u = delta of the reference (dim nu):
//
//   min   sum_{k=0}^{N-1}  (Cy s_{k+1} - ref_k)' Qy_k (Cy s_{k+1} - ref_k)  +  u_k' R_k u_k
//                         + z_quad * |e_k|^2 + zl_k' e_k                      (slacks e_k >= 0)
//   s.t.  s_{k+1} = A s_k + B u_k,            s_0 = x0
//         lg_k <= C s_k + D u_k <= ug_k       (rows in soft_rows relaxed by e_k)
//
// The constraint rows are written at stage k as (C_out A) s_k + (C_out B) u_k, i.e. they
// constrain the predicted state s_{k+1} (same convention as the Python condensed MPC, whose
// first constraint block is Phi_c[0] = C_out A and G_c[0] = C_out B).
//
// This is the same problem the Python BaseRGCController hands to OSQP in condensed form.
#pragma once

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <vector>

namespace rgc_mpc {

using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vec = Eigen::VectorXd;

struct StageQp {
    Mat A, B;             // nx x nx, nx x nu   (constant over the horizon)
    Mat Cy;               // ny x nx            tracked outputs
    std::vector<Mat> Qy;  // N blocks ny x ny   output weights
    Mat ref;              // N x ny             output references (row k -> s_{k+1})
    std::vector<Mat> R;   // N blocks nu x nu   input weights
    Mat C, D;             // ng x nx, ng x nu   general constraints
    Mat lg, ug;           // N x ng             bounds (+-inf allowed -> not enforced)
    Mat zl;               // N x ns             linear slack weights (soft rows)
    double z_quad = 0.0;  //                    quadratic slack weight
    Vec x0;               // nx                 current augmented state
};

struct SolverOptions {
    double tol = 1e-6;
    int iter_max = 100;
    std::string mode = "speed_abs";  // speed_abs | speed | balance | robust
};

struct StageQpSolution {
    int status = -1;  // HPIPM: 0 success, 1 max iter, 2 min step, 3 NaN
    int iterations = 0;
    double res_stat = 0, res_eq = 0, res_ineq = 0, res_comp = 0;
    Mat U;  // N x nu        optimal inputs u_0..u_{N-1}
    Mat X;  // (N+1) x nx    predicted states (row 0 = x0)
    Mat S;  // N x ns        slack of each soft row (lower + upper side)
    double lam_max = 0;      // max |multiplier| of the general constraints
    double tracking_cost = 0, effort_cost = 0, slack_cost = 0;
    double solve_time = 0;   // [s] HPIPM IPM only
};

class HpipmStageQpSolver {
public:
    // Dimensions are fixed for the lifetime of the solver (one solver per mode).
    HpipmStageQpSolver(int N, int nx, int nu, int ng, std::vector<int> soft_rows, SolverOptions options = {});
    ~HpipmStageQpSolver();

    HpipmStageQpSolver(const HpipmStageQpSolver&) = delete;
    HpipmStageQpSolver& operator=(const HpipmStageQpSolver&) = delete;

    StageQpSolution solve(const StageQp& qp);

    int N() const;
    int nx() const;
    int nu() const;
    int ng() const;
    int ns() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rgc_mpc
