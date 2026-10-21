#include <fstream>
#include <iostream>
#include <queue>
#include <filesystem>

#include <Eigen/Core>

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

#include "estimator/ba.hpp"

#include "asr_loc/BAResult.h"
#include "asr_loc/Frame.h"

std::queue<asr_loc::Frame> frame_msg_queue;
void FrameMsgCallback(const asr_loc::Frame& msg)
{
    frame_msg_queue.push(msg);
    if (frame_msg_queue.size() > 10) {
        frame_msg_queue.pop();
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ba_node");
    ros::NodeHandle nh;

    // subscribe frame (from tracking node)
    ros::Subscriber frame_sub = nh.subscribe("/frame", 200, FrameMsgCallback);

    // publish anchor positions
    ros::Publisher aps_pub = nh.advertise<asr_loc::BAResult>("/ba_result", 20);
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("/aps_marker", 20);

    // save results
    std::string save_path;
    nh.param<std::string>("save_path", save_path, "");

    std::filesystem::path dir = save_path;
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }

    std::string aps_save_path = save_path + "/est_aps.txt";
    std::ofstream aps_save_fs;
    aps_save_fs.open(aps_save_path, std::ios::out);
    aps_save_fs.setf(std::ios::fixed, std::ios::floatfield);
    aps_save_fs.precision(16);

    // anchor
    int num_anchors = 0;
    nh.param<int>("num_anchors", num_anchors, 0);

    // initial anchor postions
    std::vector<double> aps_val_vec;
    nh.param<std::vector<double>>("anchor_positions", aps_val_vec, std::vector<double>());

    ROS_ASSERT_MSG(aps_val_vec.size() == (num_anchors * 3), "Please check the anchor configuration!");

    std::vector<Eigen::Vector3d> initial_aps_vec;
    for (size_t i = 0; i < num_anchors; i++) {
        initial_aps_vec.emplace_back(aps_val_vec[i * 3], aps_val_vec[i * 3 + 1], aps_val_vec[i * 3 + 2]);
    }

    // for (const auto & p : initial_aps_vec) {
    //     std::cout << p.transpose() << '\n';
    // }

    // ranges between anchors
    std::vector<double> aps_range_val_vec;
    nh.param<std::vector<double>>("aps_range", aps_range_val_vec, std::vector<double>());

    ROS_ASSERT_MSG(aps_range_val_vec.size() == ((num_anchors - 3) * 3), "Please check the anchor configuration!");

    Eigen::MatrixXd aps_range_mat;
    aps_range_mat.setZero((num_anchors - 3), 3);
    for (size_t i = 0; i < (num_anchors - 3); i++) {
        aps_range_mat(i, 0) = aps_range_val_vec[i * 3];
        aps_range_mat(i, 1) = aps_range_val_vec[i * 3 + 1];
        aps_range_mat(i, 2) = aps_range_val_vec[i * 3 + 2];
    }

    // std::cout << aps_range_mat << "\n\n";

    // bias
    double bias_walk_var = 0;
    nh.param<double>("bias_walk_var", bias_walk_var, 0);

    std::vector<double> bias_vec;
    nh.param<std::vector<double>>("bias", bias_vec, std::vector<double>());

    // for (const auto val : bias_vec) {
    //     std::cout << val << " ";
    // }
    // std::cout << std::endl;

    // noise
    double noise_var = 0;
    nh.param<double>("noise_var", noise_var, 0);

    /// core part
    BA ba(initial_aps_vec, aps_range_mat, bias_vec, noise_var, bias_walk_var);

    ros::Rate rate(10);
    while (ros::ok()) {
        ros::spinOnce();

        // BA
        if (!frame_msg_queue.empty()) {
            ba.addFrame(frame_msg_queue.front());
            frame_msg_queue.pop();
        }

        ba.solve();
        const std::vector<Eigen::Vector3d> esti_aps_vec = ba.getAnchorsPositions();
        const std::vector<double> esti_bias_vec = ba.getBias();

        // publish ba result
        asr_loc::BAResult ba_result_msg;
        double stamp = ros::Time::now().toSec();
        if (std::fabs(stamp - 0) < 1e-3) {
            continue;
        }
        ba_result_msg.stamp = stamp;
        ba_result_msg.num_anchors = num_anchors;
        for (int i = 0; i < esti_aps_vec.size(); i++) {
            const Eigen::Vector3d q_i = esti_aps_vec.at(i);
            for (int j = 0; j < 3; j++) {
                ba_result_msg.aps_arr[i * 3 + j] = q_i(j);
            }
        }
        for (int i = 0; i < esti_bias_vec.size(); i++) {
            ba_result_msg.bias_arr[i] = esti_bias_vec.at(i);
        }
        aps_pub.publish(ba_result_msg);

        // publish anchor markers
        for (int i = 0; i < esti_aps_vec.size(); i++) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "world";
            marker.header.stamp = ros::Time::now();

            marker.ns = "anchors";
            marker.id = i;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;

            const Eigen::Vector3d q_i = esti_aps_vec.at(i);
            marker.pose.position.x = q_i(0);
            marker.pose.position.y = q_i(1);
            marker.pose.position.z = q_i(2);
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;

            marker.scale.x = 0.2;
            marker.scale.y = 0.2;
            marker.scale.z = 0.2;

            marker.color.r = 1.0;
            marker.color.g = 0.54;
            marker.color.b = 0.0;
            marker.color.a = 1.0;

            marker.lifetime = ros::Duration();

            marker_pub.publish(marker);
        }

        // record
        std::stringstream ss;
        ss.setf(std::ios::fixed, std::ios::floatfield);

        ss.precision(9);
        ss << stamp;
        ss.precision(16);
        for (const auto& q_i : esti_aps_vec) {
            ss << " " << q_i(0) << " " << q_i(1) << " " << q_i(2);
        }
        ss << "\n";
        aps_save_fs << ss.str();

        // ss.str("");
        // ss.precision(9);
        // ss << stamp;
        // ss.precision(16);
        // for (const auto& bias : esti_bias_vec) {
        //     ss << " " << bias;
        // }
        // ss << '\n';
        // bias_save_fs << ss.str();

        rate.sleep();
    }

    aps_save_fs.close();
    // bias_save_fs.close();

    return 0;
}