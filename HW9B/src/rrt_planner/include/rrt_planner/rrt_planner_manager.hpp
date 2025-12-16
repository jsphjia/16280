#pragma once

#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <rclcpp/rclcpp.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <memory>
#include <string>
#include <vector>

namespace rrt_planner
{

class RrtPlanningContext;

class RrtPlannerManager : public planning_interface::PlannerManager
{
public:
  RrtPlannerManager() = default;
  ~RrtPlannerManager() override = default;

  bool initialize(const moveit::core::RobotModelConstPtr& model,
                  const rclcpp::Node::SharedPtr& node,
                  const std::string& ns) override;

  bool canServiceRequest(const planning_interface::MotionPlanRequest& req) const override;

  std::string getDescription() const override { return "RrtPlannerManager"; }

  void getPlanningAlgorithms(std::vector<std::string>& algs) const override
  {
    algs = {"RRT"};
  }

  planning_interface::PlanningContextPtr getPlanningContext(
      const planning_scene::PlanningSceneConstPtr& planning_scene,
      const planning_interface::MotionPlanRequest& req,
      moveit_msgs::msg::MoveItErrorCodes& error_code) const override;

  void setPlannerConfigurations(
      const planning_interface::PlannerConfigurationMap& pcs) override
  {
    configs_ = pcs;
  }

private:
  moveit::core::RobotModelConstPtr model_;
  rclcpp::Node::SharedPtr node_;
  std::string ns_;
  planning_interface::PlannerConfigurationMap configs_;
};

}  // namespace rrt_planner
