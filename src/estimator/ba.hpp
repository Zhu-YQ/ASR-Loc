#pragma once

#include <Eigen/Core>
#include <deque>
#include <set>
#include <vector>

#include "optimization.hpp"

#include "asr_loc/Frame.h"

class BA {
private:
    std::vector<PositionVertex*> anchor_p_v_vec_;
    std::deque<PositionVertex*> tag_p_v_vec_;
    std::vector<std::deque<BiasVertex*>> bias_v_queue_vec_;

    std::set<int> basic_anchor_id_set_;

    int lastest_vertex_id_;
    int lastest_edge_id_;

    double noise_var_;
    double bias_walk_var_;

    std::vector<double> initial_bias_vec_;

    g2o::SparseOptimizer optimizer_;

public:
    BA(std::vector<Eigen::Vector3d>& aps_vec, const Eigen::MatrixXd& aps_range_mat, const std::vector<double>& initial_bias_vec, double noise_var, double bias_walk_var)
        : anchor_p_v_vec_()
        , tag_p_v_vec_()
        , bias_v_queue_vec_()
        , basic_anchor_id_set_()
        , lastest_vertex_id_(-1)
        , lastest_edge_id_(-1)
        , noise_var_(noise_var)
        , bias_walk_var_(bias_walk_var)
        , initial_bias_vec_(initial_bias_vec)
        , optimizer_()
    {
        basic_anchor_id_set_.insert(0);
        basic_anchor_id_set_.insert(1);
        basic_anchor_id_set_.insert(2);

        using BlockSolver = g2o::BlockSolverX;
        // using LinearSolverType = g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;
        using LinearSolverType = g2o::LinearSolverCSparse<BlockSolver::PoseMatrixType>;

        auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolverType>()));
        // auto solver = new g2o::OptimizationAlgorithmDogleg(g2o::make_unique<BlockSolver>(g2o::make_unique<LinearSolverType>()));
        optimizer_.setAlgorithm(solver);

        for (int i = 0; i < aps_vec.size(); i++) {
            auto v = new PositionVertex();
            v->setId(++lastest_vertex_id_);
            v->setEstimate(aps_vec.at(i));

            // exclude basic anchors
            if (basic_anchor_id_set_.count(i)) {
                v->setFixed(true);
            }

            anchor_p_v_vec_.push_back(v);
            optimizer_.addVertex(v);
        }

        for (int i = 0; i < aps_vec.size(); i++) {
            bias_v_queue_vec_.push_back(std::deque<BiasVertex*>());
        }

        for (int i = 0; i < aps_range_mat.rows(); i++) {
            for (int j = 0; j < aps_range_mat.cols(); j++) {
                auto ap_ij_v = new RangeEdge();
                ap_ij_v->setId(++lastest_edge_id_);
                ap_ij_v->setVertex(0, anchor_p_v_vec_.at(i + 3));
                ap_ij_v->setVertex(1, anchor_p_v_vec_.at(j));

                ap_ij_v->setMeasurement(aps_range_mat(i, j));
                ap_ij_v->setInformation(1 * (1.0 / noise_var_) * Eigen::Matrix<double, 1, 1>::Identity());
                optimizer_.addEdge(ap_ij_v);
            }
        }
    }

    ~BA() = default;

    void addFrame(const asr_loc::Frame& frame)
    {
        // add one tag vertex
        auto tp_v = new PositionVertex();
        tp_v->setId(++lastest_vertex_id_);
        const Eigen::Vector3d initial_tp(frame.p_esti[0], frame.p_esti[1], frame.p_esti[2]);
        tp_v->setEstimate(initial_tp);
        optimizer_.addVertex(tp_v);
        tag_p_v_vec_.push_back(tp_v);

        // add m bias vertices
        for (int i = 0; i < anchor_p_v_vec_.size(); i++) {
            // for (auto& bias_v_queue: bias_v_queue_vec_){
            auto& bias_v_queue = bias_v_queue_vec_.at(i);
            auto b_v = new BiasVertex();
            b_v->setId(++lastest_vertex_id_);
            const double initial_bias = bias_v_queue.empty() ? initial_bias_vec_.at(i) : bias_v_queue.back()->estimate();
            b_v->setEstimate(initial_bias);
            optimizer_.addVertex(b_v);
            bias_v_queue.push_back(b_v);
        }

        // add m range edges
        for (int i = 0; i < anchor_p_v_vec_.size(); i++) {
            auto range_e = new BiasedRangeEdge();
            range_e->setId(++lastest_edge_id_);
            range_e->setVertex(0, tag_p_v_vec_.back());
            range_e->setVertex(1, anchor_p_v_vec_.at(i));
            range_e->setVertex(2, bias_v_queue_vec_.at(i).back());

            // auto range_e = new RangeEdge();
            // range_e->setId(++lastest_edge_id_);
            // range_e->setVertex(0, tag_p_v_vec_.back());
            // range_e->setVertex(1, anchor_p_v_vec_.at(i));

            range_e->setMeasurement(frame.dis_arr[i]);
            range_e->setInformation((1.0 / noise_var_) * Eigen::Matrix<double, 1, 1>::Identity());

            // auto robust_kernel = new g2o::RobustKernelHuber();
            // auto robust_kernel = new g2o::RobustKernelDCS();
            auto robust_kernel = new g2o::RobustKernelGemanMcClure();
            robust_kernel->setDelta(0.01);
            range_e->setRobustKernel(robust_kernel);

            optimizer_.addEdge(range_e);
        }

        // add m bias random walk edges
        if (bias_v_queue_vec_.at(0).size() < 2) {
            return;
        }
        for (auto& bias_v_queue : bias_v_queue_vec_) {
            // for (int i = 0; i < anchor_p_v_vec_.size(); i++) {
            auto bias_walk_e = new BiasWalkEdge();
            bias_walk_e->setId(++lastest_edge_id_);

            // std::deque<BiasVertex*> b_v_queue = bias_v_queue_vec_.at(i);
            bias_walk_e->setVertex(0, bias_v_queue.at(bias_v_queue.size() - 2));
            bias_walk_e->setVertex(1, bias_v_queue.at(bias_v_queue.size() - 1));

            bias_walk_e->setInformation((1.0 / bias_walk_var_) * Eigen::Matrix<double, 1, 1>::Identity());

            optimizer_.addEdge(bias_walk_e);
        }
    }

    void solve()
    {
        if (tag_p_v_vec_.size() < 900) {
            return;
        }
        optimizer_.initializeOptimization();
        optimizer_.optimize(30);
    }

    std::vector<Eigen::Vector3d> getAnchorsPositions() const
    {
        std::vector<Eigen::Vector3d> aps_esti_vec;
        for (auto v : anchor_p_v_vec_) {
            aps_esti_vec.push_back(v->estimate());
        }
        return aps_esti_vec;
    }

    std::vector<double> getBias() const
    {
        std::vector<double> bias_vec;
        for (auto& b_v_queue : bias_v_queue_vec_) {
            double bias = 0;
            if (!b_v_queue.empty()) {
                bias = b_v_queue.back()->estimate();
            }
            bias_vec.push_back(bias);
        }
        return bias_vec;
    }
};
