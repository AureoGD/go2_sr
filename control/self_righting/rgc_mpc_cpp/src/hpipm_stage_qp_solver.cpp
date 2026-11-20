// HPIPM (OCP QP, interior point) solver for the RGC stage-wise MPC QP. See stage_qp.hpp.
//
// HPIPM layout used here (s_0 is eliminated: nx[0] = 0):
//   stage 0     : u_0                      b_0 = A x0,   lg_0 - C x0 <= D u_0 <= ug_0 - C x0
//   stage k<N   : (s_k, u_k)               s_{k+1} = A s_k + B u_k,  lg_k <= C s_k + D u_k <= ug_k
//   stage N     : s_N                      cost only
// Cost (HPIPM uses 1/2 x'Hx): Q_k = 2 Cy' Qy_{k-1} Cy, q_k = -2 Cy' Qy_{k-1} ref_{k-1}, R_k = 2 R_k,
// slacks: Zl = Zu = 2 z_quad, zl = zu = zl_k (one slack per side, at most one is active).
#include "rgc_mpc/stage_qp.hpp"

extern "C" {
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp_ipm.h>
#include <hpipm_d_ocp_qp_sol.h>
}

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace rgc_mpc {

namespace {

constexpr double kInfThreshold = 1e15;  // |bound| above this (or non-finite) -> not enforced

hpipm_mode parse_mode(const std::string& mode) {
    if (mode == "speed_abs") return SPEED_ABS;
    if (mode == "speed") return SPEED;
    if (mode == "balance") return BALANCE;
    if (mode == "robust") return ROBUST;
    throw std::invalid_argument("HpipmStageQpSolver: unknown mode '" + mode + "'");
}

void* checked_malloc(hpipm_size_t size) {
    void* p = std::malloc(size);
    if (p == nullptr) throw std::bad_alloc();
    return p;
}

void check_shape(const Mat& m, Eigen::Index rows, Eigen::Index cols, const char* name) {
    if (m.rows() != rows || m.cols() != cols) {
        throw std::invalid_argument(std::string("HpipmStageQpSolver: ") + name + " has shape (" +
                                    std::to_string(m.rows()) + ", " + std::to_string(m.cols()) + "), expected (" +
                                    std::to_string(rows) + ", " + std::to_string(cols) + ")");
    }
}

// HPIPM setters take non-const pointers but only read from them
inline double* ptr(const Mat& m) { return const_cast<double*>(m.data()); }
inline double* ptr(const Vec& v) { return const_cast<double*>(v.data()); }

}  // namespace

struct HpipmStageQpSolver::Impl {
    int N, nx, nu, ng, ns;
    std::vector<int> soft_rows;
    SolverOptions options;

    std::vector<int> v_nx, v_nu, v_nbx, v_nbu, v_ng, v_ns;

    void *dim_mem = nullptr, *qp_mem = nullptr, *sol_mem = nullptr, *arg_mem = nullptr, *ws_mem = nullptr;
    d_ocp_qp_dim dim;
    d_ocp_qp qp;
    d_ocp_qp_sol sol;
    d_ocp_qp_ipm_arg arg;
    d_ocp_qp_ipm_ws ws;

    // scratch buffers (allocated once)
    Mat Qk, Rk;
    Vec qk, bk, lgk, ugk, lg_mask, ug_mask, Zvec, zvec, buf_u, buf_x, buf_s, buf_lam;

    ~Impl() {
        std::free(ws_mem);
        std::free(arg_mem);
        std::free(sol_mem);
        std::free(qp_mem);
        std::free(dim_mem);
    }
};

HpipmStageQpSolver::HpipmStageQpSolver(int N, int nx, int nu, int ng, std::vector<int> soft_rows,
                                       SolverOptions options)
    : impl_(std::make_unique<Impl>()) {
    if (N <= 0 || nx <= 0 || nu <= 0 || ng < 0) throw std::invalid_argument("HpipmStageQpSolver: invalid dimensions");
    for (int r : soft_rows) {
        if (r < 0 || r >= ng) throw std::invalid_argument("HpipmStageQpSolver: soft row out of range");
    }

    Impl& m = *impl_;
    m.N = N;
    m.nx = nx;
    m.nu = nu;
    m.ng = ng;
    m.ns = static_cast<int>(soft_rows.size());
    m.soft_rows = std::move(soft_rows);
    m.options = std::move(options);

    m.v_nx.assign(N + 1, nx);
    m.v_nx[0] = 0;  // s_0 is known: eliminated
    m.v_nu.assign(N + 1, nu);
    m.v_nu[N] = 0;
    m.v_nbx.assign(N + 1, 0);
    m.v_nbu.assign(N + 1, 0);
    m.v_ng.assign(N + 1, ng);
    m.v_ng[N] = 0;
    m.v_ns.assign(N + 1, m.ns);
    m.v_ns[N] = 0;

    m.dim_mem = checked_malloc(d_ocp_qp_dim_memsize(N));
    d_ocp_qp_dim_create(N, &m.dim, m.dim_mem);
    d_ocp_qp_dim_set_all(m.v_nx.data(), m.v_nu.data(), m.v_nbx.data(), m.v_nbu.data(), m.v_ng.data(), m.v_ns.data(),
                         &m.dim);

    m.qp_mem = checked_malloc(d_ocp_qp_memsize(&m.dim));
    d_ocp_qp_create(&m.dim, &m.qp, m.qp_mem);
    d_ocp_qp_set_all_zero(&m.qp);  // S, r, slack bounds (lls = lus = 0) stay zero

    m.sol_mem = checked_malloc(d_ocp_qp_sol_memsize(&m.dim));
    d_ocp_qp_sol_create(&m.dim, &m.sol, m.sol_mem);

    m.arg_mem = checked_malloc(d_ocp_qp_ipm_arg_memsize(&m.dim));
    d_ocp_qp_ipm_arg_create(&m.dim, &m.arg, m.arg_mem);
    d_ocp_qp_ipm_arg_set_default(parse_mode(m.options.mode), &m.arg);
    double tol = m.options.tol;
    int iter_max = m.options.iter_max;
    d_ocp_qp_ipm_arg_set_iter_max(&iter_max, &m.arg);
    d_ocp_qp_ipm_arg_set_tol_stat(&tol, &m.arg);
    d_ocp_qp_ipm_arg_set_tol_eq(&tol, &m.arg);
    d_ocp_qp_ipm_arg_set_tol_ineq(&tol, &m.arg);
    d_ocp_qp_ipm_arg_set_tol_comp(&tol, &m.arg);

    m.ws_mem = checked_malloc(d_ocp_qp_ipm_ws_memsize(&m.dim, &m.arg));
    d_ocp_qp_ipm_ws_create(&m.dim, &m.arg, &m.ws, m.ws_mem);

    // soft rows: same general-constraint rows at every stage (no box constraints -> nb = 0)
    if (m.ns > 0) {
        for (int k = 0; k < N; ++k) d_ocp_qp_set_idxs(k, m.soft_rows.data(), &m.qp);
    }

    m.Qk.resize(nx, nx);
    m.Rk.resize(nu, nu);
    m.qk.resize(nx);
    m.bk.resize(nx);
    m.lgk.resize(ng);
    m.ugk.resize(ng);
    m.lg_mask.resize(ng);
    m.ug_mask.resize(ng);
    m.Zvec.resize(m.ns);
    m.zvec.resize(m.ns);
    m.buf_u.resize(nu);
    m.buf_x.resize(nx);
    m.buf_s.resize(m.ns);
    m.buf_lam.resize(ng);
}

HpipmStageQpSolver::~HpipmStageQpSolver() = default;

int HpipmStageQpSolver::N() const { return impl_->N; }
int HpipmStageQpSolver::nx() const { return impl_->nx; }
int HpipmStageQpSolver::nu() const { return impl_->nu; }
int HpipmStageQpSolver::ng() const { return impl_->ng; }
int HpipmStageQpSolver::ns() const { return impl_->ns; }

StageQpSolution HpipmStageQpSolver::solve(const StageQp& p) {
    Impl& m = *impl_;
    const int N = m.N, nx = m.nx, nu = m.nu, ng = m.ng, ns = m.ns;
    const auto ny = p.Cy.rows();

    // ------------------------------------------------------------ checks
    check_shape(p.A, nx, nx, "A");
    check_shape(p.B, nx, nu, "B");
    if (p.Cy.cols() != nx) throw std::invalid_argument("HpipmStageQpSolver: Cy must have nx columns");
    if (static_cast<int>(p.Qy.size()) != N || static_cast<int>(p.R.size()) != N)
        throw std::invalid_argument("HpipmStageQpSolver: Qy and R need N blocks");
    check_shape(p.ref, N, ny, "ref");
    check_shape(p.C, ng, nx, "C");
    check_shape(p.D, ng, nu, "D");
    check_shape(p.lg, N, ng, "lg");
    check_shape(p.ug, N, ng, "ug");
    check_shape(p.zl, N, ns, "zl");
    if (p.x0.size() != nx) throw std::invalid_argument("HpipmStageQpSolver: x0 must have nx entries");

    // ------------------------------------------------------------ dynamics
    m.bk = p.A * p.x0;
    d_ocp_qp_set_B(0, ptr(p.B), &m.qp);
    d_ocp_qp_set_b(0, ptr(m.bk), &m.qp);
    m.bk.setZero();
    for (int k = 1; k < N; ++k) {
        d_ocp_qp_set_A(k, ptr(p.A), &m.qp);
        d_ocp_qp_set_B(k, ptr(p.B), &m.qp);
        d_ocp_qp_set_b(k, ptr(m.bk), &m.qp);
    }

    // ------------------------------------------------------------ cost
    for (int k = 0; k <= N; ++k) {
        if (k >= 1) {  // output tracking on s_k with block k-1
            check_shape(p.Qy[k - 1], ny, ny, "Qy block");
            m.Qk.noalias() = 2.0 * p.Cy.transpose() * p.Qy[k - 1] * p.Cy;
            m.qk.noalias() = -2.0 * p.Cy.transpose() * (p.Qy[k - 1] * p.ref.row(k - 1).transpose());
            d_ocp_qp_set_Q(k, ptr(m.Qk), &m.qp);
            d_ocp_qp_set_q(k, ptr(m.qk), &m.qp);
        }
        if (k < N) {
            check_shape(p.R[k], nu, nu, "R block");
            m.Rk = 2.0 * p.R[k];
            d_ocp_qp_set_R(k, ptr(m.Rk), &m.qp);
        }
    }

    // ------------------------------------------------------------ general constraints
    if (ng > 0) {
        for (int k = 0; k < N; ++k) {
            m.lgk = p.lg.row(k).transpose();
            m.ugk = p.ug.row(k).transpose();
            if (k == 0) {  // s_0 known: move C x0 to the bounds
                Vec cx0 = p.C * p.x0;
                m.lgk -= cx0;
                m.ugk -= cx0;
            } else {
                d_ocp_qp_set_C(k, ptr(p.C), &m.qp);
            }
            d_ocp_qp_set_D(k, ptr(p.D), &m.qp);

            for (int i = 0; i < ng; ++i) {
                const bool lo = std::isfinite(p.lg(k, i)) && std::abs(p.lg(k, i)) < kInfThreshold;
                const bool up = std::isfinite(p.ug(k, i)) && std::abs(p.ug(k, i)) < kInfThreshold;
                m.lg_mask[i] = lo ? 1.0 : 0.0;
                m.ug_mask[i] = up ? 1.0 : 0.0;
                if (!lo) m.lgk[i] = 0.0;
                if (!up) m.ugk[i] = 0.0;
            }
            d_ocp_qp_set_lg(k, ptr(m.lgk), &m.qp);
            d_ocp_qp_set_ug(k, ptr(m.ugk), &m.qp);
            d_ocp_qp_set_lg_mask(k, ptr(m.lg_mask), &m.qp);
            d_ocp_qp_set_ug_mask(k, ptr(m.ug_mask), &m.qp);

            if (ns > 0) {
                m.Zvec.setConstant(2.0 * p.z_quad);
                m.zvec = p.zl.row(k).transpose();
                d_ocp_qp_set_Zl(k, ptr(m.Zvec), &m.qp);
                d_ocp_qp_set_Zu(k, ptr(m.Zvec), &m.qp);
                d_ocp_qp_set_zl(k, ptr(m.zvec), &m.qp);
                d_ocp_qp_set_zu(k, ptr(m.zvec), &m.qp);
            }
        }
    }

    // ------------------------------------------------------------ solve
    const auto t0 = std::chrono::steady_clock::now();
    d_ocp_qp_ipm_solve(&m.qp, &m.sol, &m.arg, &m.ws);
    const auto t1 = std::chrono::steady_clock::now();

    StageQpSolution out;
    out.solve_time = std::chrono::duration<double>(t1 - t0).count();
    d_ocp_qp_ipm_get_status(&m.ws, &out.status);
    d_ocp_qp_ipm_get_iter(&m.ws, &out.iterations);
    d_ocp_qp_ipm_get_max_res_stat(&m.ws, &out.res_stat);
    d_ocp_qp_ipm_get_max_res_eq(&m.ws, &out.res_eq);
    d_ocp_qp_ipm_get_max_res_ineq(&m.ws, &out.res_ineq);
    d_ocp_qp_ipm_get_max_res_comp(&m.ws, &out.res_comp);

    // ------------------------------------------------------------ solution
    out.U.resize(N, nu);
    out.X.resize(N + 1, nx);
    out.S.resize(N, ns);
    out.X.row(0) = p.x0.transpose();

    for (int k = 0; k < N; ++k) {
        d_ocp_qp_sol_get_u(k, &m.sol, m.buf_u.data());
        out.U.row(k) = m.buf_u.transpose();

        d_ocp_qp_sol_get_x(k + 1, &m.sol, m.buf_x.data());
        out.X.row(k + 1) = m.buf_x.transpose();

        if (ns > 0) {
            d_ocp_qp_sol_get_sl(k, &m.sol, m.buf_s.data());
            Vec s = m.buf_s;
            d_ocp_qp_sol_get_su(k, &m.sol, m.buf_s.data());
            s += m.buf_s;
            out.S.row(k) = s.transpose();
        }

        if (ng > 0) {
            d_ocp_qp_sol_get_lam_lg(k, &m.sol, m.buf_lam.data());
            out.lam_max = std::max(out.lam_max, m.buf_lam.cwiseAbs().maxCoeff());
            d_ocp_qp_sol_get_lam_ug(k, &m.sol, m.buf_lam.data());
            out.lam_max = std::max(out.lam_max, m.buf_lam.cwiseAbs().maxCoeff());
        }
    }

    // ------------------------------------------------------------ cost terms (original scale)
    for (int k = 0; k < N; ++k) {
        const Vec e = p.Cy * out.X.row(k + 1).transpose() - p.ref.row(k).transpose();
        out.tracking_cost += e.dot(p.Qy[k] * e);
        const Vec u = out.U.row(k).transpose();
        out.effort_cost += u.dot(p.R[k] * u);
        if (ns > 0) {
            const Vec s = out.S.row(k).transpose();
            out.slack_cost += p.z_quad * s.squaredNorm() + p.zl.row(k).dot(s.transpose());
        }
    }

    return out;
}

}  // namespace rgc_mpc
