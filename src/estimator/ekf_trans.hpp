#pragma once

#include <cmath>
#include <queue>

#include "optimization.hpp"

class EKFtrans {
private:
    const int ROBOT_STATE_DIM_ = 9;
    const int ROBOT_SPACE_DIM_ = 3;

    const int n_;
    const int m_;

    Eigen::VectorXd x_;
    Eigen::MatrixXd P_;

    const double NLOS_THRESH_;

    Eigen::MatrixXd calHi(const Eigen::Vector3d& q_i) const
    {
        const Eigen::Vector3d p = getTagPosition();

        Eigen::Vector3d dp = p - q_i;
        double distance = dp.norm();

        Eigen::MatrixXd H_i;
        H_i.setZero(1, n_);
        H_i(0, 0) = dp(0) / distance;
        H_i(0, 3) = dp(1) / distance;
        H_i(0, 6) = dp(2) / distance;
        return H_i;
    }

    double calYHati(const Eigen::Vector3d& q_i) const
    {
        const Eigen::Vector3d p = getTagPosition();
        return (p - q_i).norm();
    }

public:
    EKFtrans(int num_anchors, double nlos_thresh,
        double initial_robot_px_var, double initial_robot_py_var, double initial_robot_pz_var,
        double initial_robot_v_var, double initial_robot_a_var)
        : n_(ROBOT_STATE_DIM_)
        , m_(num_anchors)
        , NLOS_THRESH_(nlos_thresh)
    {

        x_.setZero(n_, 1);
        P_.setZero(n_, n_);

        P_(0, 0) = initial_robot_px_var;
        P_(3, 3) = initial_robot_py_var;
        P_(6, 6) = initial_robot_pz_var;

        P_(1, 1) = P_(4, 4) = P_(7, 7) = initial_robot_v_var;
        P_(2, 2) = P_(5, 5) = P_(8, 8) = initial_robot_a_var;
    }

    ~EKFtrans() = default;

    static Eigen::Vector3d LocateOnce(const Eigen::VectorXd& y, const std::vector<Eigen::Vector3d>& aps_vec,
        const Eigen::Vector3d& p0 = Eigen::Vector3d::Ones(), const std::vector<double>& bias_vec = {})
    {

        auto* v = new PositionVertex();
        v->setId(0);
        v->setEstimate(p0);

        using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>>;
        using LinearSolverType = g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;
        auto solver = new g2o::OptimizationAlgorithmDogleg(g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        optimizer.addVertex(v);

        // add edges
        for (int i = 0; i < aps_vec.size(); i++) {
            double bias = 0;
            if (!bias_vec.empty()) {
                bias = bias_vec.at(i);
            }

            auto edge = new OptimLocEdge(aps_vec.at(i), bias);
            edge->setId(i);
            edge->setVertex(0, v);

            edge->setMeasurement(y(i));
            edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());

            auto robust_kernel = new g2o::RobustKernelGemanMcClure();
            robust_kernel->setDelta(0.5);
            edge->setRobustKernel(robust_kernel);

            optimizer.addEdge(edge);
        }

        optimizer.initializeOptimization();
        optimizer.optimize(20);

        return v->estimate();
    }

    Eigen::Vector3d getTagPosition() const
    {
        return Eigen::Vector3d(x_(0), x_(3), x_(6));
    }

    Eigen::MatrixXd getTagPositionCov() const
    {
        Eigen::MatrixXd P_pos;
        P_pos.setZero(ROBOT_SPACE_DIM_, ROBOT_SPACE_DIM_);
        P_pos(0, 0) = P_(0, 0);
        P_pos(0, 1) = P_(0, 3);
        P_pos(0, 2) = P_(0, 6);
        P_pos(1, 0) = P_(3, 0);
        P_pos(1, 1) = P_(3, 3);
        P_pos(1, 2) = P_(3, 6);
        P_pos(2, 0) = P_(6, 0);
        P_pos(2, 1) = P_(6, 3);
        P_pos(2, 2) = P_(6, 6);
        return P_pos;
    }

    std::vector<double> getTagPositionBound() const
    {
        return { 3.0 * std::sqrt(P_(0, 0)), 3.0 * std::sqrt(P_(3, 3)), 3.0 * std::sqrt(P_(6, 6)) };
    }

    void initTagPosition(const Eigen::VectorXd& y, double tag_height, const std::vector<Eigen::Vector3d>& aps_vec)
    {
        Eigen::Vector3d p0 = LocateOnce(y, aps_vec);
        x_(0) = p0(0);
        x_(3) = p0(1);
        // x_(6) = tag_height;
        x_(6) = p0(2);
    }

    void predict(double dt, double motion_robot_jx_var, double motion_robot_jy_var, double motion_robot_jz_var)
    {
        // F
        Eigen::MatrixXd F;
        F.setIdentity(n_, n_);
        F(0, 1) = F(3, 4) = F(6, 7) = dt;
        F(0, 2) = F(3, 5) = F(6, 8) = 0.5 * dt * dt;
        F(1, 2) = F(4, 5) = F(7, 8) = dt;

        // G
        Eigen::MatrixXd G;
        G.setZero(n_, ROBOT_SPACE_DIM_);
        G(0, 0) = G(3, 1) = G(6, 2) = (1.0 / 6) * dt * dt * dt;
        G(1, 0) = G(4, 1) = G(7, 2) = 0.5 * dt * dt;
        G(2, 0) = G(5, 1) = G(8, 2) = dt;

        // Q
        Eigen::MatrixXd Q;
        Q.setZero(ROBOT_SPACE_DIM_, ROBOT_SPACE_DIM_);
        Q.diagonal() << motion_robot_jx_var, motion_robot_jy_var, motion_robot_jz_var;

        x_ = F * x_;
        P_ = F * P_ * F.transpose() + G * Q * G.transpose();
    }

    void update_seq(const Eigen::VectorXd& y, double meas_noise_var, const std::vector<Eigen::Vector3d>& aps_vec)
    {
        double y_hat_i = 0;
        double e_i = 0;
        double S = 0;

        Eigen::MatrixXd H_i, K_i, I;
        I.setIdentity(n_, n_);

        for (int i = 0; i < m_; i++) {
            y_hat_i = calYHati(aps_vec.at(i));
            e_i = y(i) - y_hat_i;

            // nlos
            if (std::abs(e_i) > NLOS_THRESH_) {
                continue;
            }

            H_i = calHi(aps_vec.at(i));

            S = (H_i * P_ * H_i.transpose())(0, 0) + meas_noise_var;
            K_i = P_ * H_i.transpose() / S;

            x_ += K_i * Eigen::Matrix<double, 1, 1>(e_i);
            P_ = (I - K_i * H_i) * P_;
        }
    }
};
