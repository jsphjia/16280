#include <rrt_planner/rrt_planner_manager.hpp>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/robot_model/robot_model.h>
#include <rclcpp/rclcpp.hpp>
#include <pluginlib/class_list_macros.hpp>
#include "rrt_planning_context.cpp"

namespace rrt_planner
{

bool RrtPlannerManager::initialize(const moveit::core::RobotModelConstPtr& model,
                                   const rclcpp::Node::SharedPtr& node,
                                   const std::string& ns)
{
  model_ = model;
  node_  = node;
  ns_    = ns;
  return static_cast<bool>(model_);
}

bool RrtPlannerManager::canServiceRequest(const planning_interface::MotionPlanRequest& req) const
{
  if (!model_ || !model_->hasJointModelGroup(req.group_name))
    return false;
  return !req.goal_constraints.empty();
}

planning_interface::PlanningContextPtr RrtPlannerManager::getPlanningContext(
    const planning_scene::PlanningSceneConstPtr& planning_scene,
    const planning_interface::MotionPlanRequest& req,
    moveit_msgs::msg::MoveItErrorCodes& error_code) const
{
  error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;

  auto context = std::make_shared<RrtPlanningContext>(
      "RRT", req.group_name, model_, planning_scene);

  context->setMotionPlanRequest(req);
  return context;
}

} // namespace rrt_planner

PLUGINLIB_EXPORT_CLASS(rrt_planner::RrtPlannerManager, planning_interface::PlannerManager)
