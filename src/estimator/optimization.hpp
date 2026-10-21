#ifndef OPTIMIZATION_HPP
#define OPTIMIZATION_HPP

#include <Eigen/Core>
#include <g2o/core/base_vertex.h>

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_unary_edge.h>

#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>

#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <g2o/core/robust_kernel_impl.h>

// for tag and anchor
class PositionVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PositionVertex() = default;
    ~PositionVertex() = default;

    void setToOriginImpl() override
    {
        // Note: (0, 0, 0) is the singularity of the Jacobian
        _estimate << 0.5, 0.5, 0.5;
    }

    void oplusImpl(const double* update) override
    {
        _estimate += Eigen::Vector3d(update);
    }

    bool read(std::istream& in) override { return false; }
    bool write(std::ostream& out) const override { return false; }
};

class BiasVertex : public g2o::BaseVertex<1, double> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BiasVertex() = default;
    ~BiasVertex() = default;

    void setToOriginImpl() override
    {
        _estimate = 0;
    }

    void oplusImpl(const double* update) override
    {
        _estimate += update[0];
    }

    bool read(std::istream& in) override { return false; }
    bool write(std::ostream& out) const override { return false; }
};

// for least squares localization
class OptimLocEdge : public g2o::BaseUnaryEdge<1, double, PositionVertex> {
private:
    Eigen::Vector3d p_known_;
    double bias_;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit OptimLocEdge(const Eigen::Vector3d& p_known, double bias = 0)
        : p_known_(p_known)
        , bias_(bias)
    {
    }

    void computeError() override
    {
        const auto* v = dynamic_cast<const PositionVertex*>(_vertices[0]);
        const Eigen::Vector3d p = v->estimate();
        _error(0, 0) = _measurement - (1 + bias_) * (p - p_known_).norm();
    }

    void linearizeOplus() override
    {
        const auto* v = dynamic_cast<const PositionVertex*>(_vertices[0]);
        const Eigen::Vector3d p = v->estimate();

        const Eigen::Vector3d difference = p - p_known_;
        const double distance = difference.norm();

        for (int i = 0; i < 3; i++) {
            _jacobianOplusXi(0, i) = -(1 + bias_) * difference[i] / distance;
        }
    }

    bool read(std::istream& in) override { return false; }
    bool write(std::ostream& out) const override { return false; }
};

// for ba, without bias
class RangeEdge : public g2o::BaseBinaryEdge<1, double, PositionVertex, PositionVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RangeEdge() = default;
    ~RangeEdge() = default;

    void computeError() override
    {
        const auto* p1_v = dynamic_cast<const PositionVertex*>(_vertices[0]);
        const Eigen::Vector3d p1 = p1_v->estimate();

        const auto* p2_v = dynamic_cast<const PositionVertex*>(_vertices[1]);
        const Eigen::Vector3d p2 = p2_v->estimate();

        _error(0, 0) = _measurement - (p1 - p2).norm();
    }

    void linearizeOplus() override
    {
        const auto* p1_v = dynamic_cast<const PositionVertex*>(_vertices[0]);
        const Eigen::Vector3d p1 = p1_v->estimate();

        const auto* p2_v = dynamic_cast<const PositionVertex*>(_vertices[1]);
        const Eigen::Vector3d p2 = p2_v->estimate();

        const Eigen::Vector3d difference = p1 - p2;
        const double distance = difference.norm();

        for (int i = 0; i < 3; i++) {
            _jacobianOplusXi(0, i) = -difference[i] / distance;
            _jacobianOplusXj(0, i) = difference[i] / distance;
        }
    }

    bool read(std::istream& in) override { return false; }
    bool write(std::ostream& out) const override { return false; }
};

// for ba, with bias
class BiasedRangeEdge : public g2o::BaseMultiEdge<1, double> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BiasedRangeEdge() { resize(3); }
    ~BiasedRangeEdge() = default;

    void computeError() override
    {
        const auto* tp_v = dynamic_cast<const PositionVertex*>(_vertices[0]);
        const Eigen::Vector3d tp = tp_v->estimate();

        const auto* ap_v = dynamic_cast<const PositionVertex*>(_vertices[1]);
        const Eigen::Vector3d ap = ap_v->estimate();

        const auto* b_v = dynamic_cast<const BiasVertex*>(_vertices[2]);
        const double bias = b_v->estimate();

        _error(0, 0) = _measurement - (1 + bias) * (tp - ap).norm();
    }

    void linearizeOplus() override
    {
        const auto* tp_v = dynamic_cast<const PositionVertex*>(_vertices[0]);
        const Eigen::Vector3d tp = tp_v->estimate();

        const auto* ap_v = dynamic_cast<const PositionVertex*>(_vertices[1]);
        const Eigen::Vector3d ap = ap_v->estimate();

        const auto* b_v = dynamic_cast<const BiasVertex*>(_vertices[2]);
        const double bias = b_v->estimate();

        const Eigen::Vector3d difference = tp - ap;
        const double distance = difference.norm();

        for (int i = 0; i < 3; i++) {
            (_jacobianOplus[0])(0, i) = -(1 + bias) * difference[i] / distance;
            (_jacobianOplus[1])(0, i) = (1 + bias) * difference[i] / distance;
        }
        (_jacobianOplus[2])(0, 0) = -distance;
    }

    bool read(std::istream& in) override { return false; }
    bool write(std::ostream& out) const override { return false; }
};

class BiasWalkEdge : public g2o::BaseBinaryEdge<1, double, BiasVertex, BiasVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BiasWalkEdge() = default;
    ~BiasWalkEdge() = default;

    void computeError() override
    {
        const auto* last_b_v = dynamic_cast<const BiasVertex*>(_vertices[0]);
        const double last_bias = last_b_v->estimate();

        const auto* new_b_v = dynamic_cast<const BiasVertex*>(_vertices[1]);
        const double new_bias = new_b_v->estimate();

        _error(0, 0) = new_bias - last_bias;
    }

    void linearizeOplus() override
    {
        _jacobianOplusXi(0, 0) = -1;
        _jacobianOplusXj(0, 0) = 1;
    }

    bool read(std::istream& in) override { return false; }
    bool write(std::ostream& out) const override { return false; }
};

class AnchorInitEdge : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, PositionVertex, PositionVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    AnchorInitEdge() = default;
    ~AnchorInitEdge() = default;

    void computeError() override
    {
        const auto* p1_v = dynamic_cast<const PositionVertex*>(_vertices[0]);
        const Eigen::Vector3d p1 = p1_v->estimate();

        const auto* p2_v = dynamic_cast<const PositionVertex*>(_vertices[1]);
        const Eigen::Vector3d p2 = p2_v->estimate();

        _error = p1 - p2;
    }

    void linearizeOplus() override
    {
        _jacobianOplusXi = Eigen::Matrix3d::Identity();
        _jacobianOplusXj = -Eigen::Matrix3d::Identity();
    }

    bool read(std::istream& in) override { return false; }
    bool write(std::ostream& out) const override { return false; }
};

#endif