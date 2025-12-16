#include <moveit/planning_interface/planning_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/kinematic_constraints/utils.h>

#include <random>
#include <limits>
#include <chrono>
#include <unordered_map>
#include <algorithm>

namespace rrt_planner
{

struct Node
{
  std::vector<double> q;
  int parent = -1;
};

class RrtPlanningContext : public planning_interface::PlanningContext
{
public:
  RrtPlanningContext(const std::string& name,
                     const std::string& group,
                     const moveit::core::RobotModelConstPtr& model,
                     const planning_scene::PlanningSceneConstPtr& scene)
  : planning_interface::PlanningContext(name, group)
  , model_(model)
  , scene_(scene)
  {}

  bool terminate() override { return false; }
  void clear() override {}

  bool solve(planning_interface::MotionPlanResponse& res) override
  {
    const auto t0 = std::chrono::steady_clock::now();

    const auto* jmg = model_->getJointModelGroup(getGroupName());
    if (!jmg)
    {
      res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
      return false;
    }

    moveit::core::RobotState start_state = scene_->getCurrentState();
    if (!request_.start_state.is_diff)
      moveit::core::robotStateMsgToRobotState(request_.start_state, start_state);
    start_state.update();

    std::vector<double> q_start, q_goal;
    start_state.copyJointGroupPositions(jmg, q_start);
    q_goal = q_start;

    if (request_.goal_constraints.empty() ||
        request_.goal_constraints[0].joint_constraints.empty())
    {
      res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
      return false;
    }

    const auto& var_names = jmg->getVariableNames();
    std::unordered_map<std::string, std::size_t> name_to_idx;
    for (std::size_t i = 0; i < var_names.size(); ++i) name_to_idx[var_names[i]] = i;

    for (const auto& jc : request_.goal_constraints[0].joint_constraints)
    {
      auto it = name_to_idx.find(jc.joint_name);
      if (it != name_to_idx.end())
        q_goal[it->second] = jc.position;
    }

    std::vector<moveit::core::VariableBounds> v_bounds;
    v_bounds.reserve(var_names.size());
    for (const auto& n : var_names) v_bounds.push_back(model_->getVariableBounds(n));
    const auto var_count = v_bounds.size();

    auto clampToBounds = [&](std::vector<double>& q)
    {
      for (size_t i = 0; i < q.size() && i < var_count; ++i)
      {
        const auto& b = v_bounds[i];
        if (b.position_bounded_)
        {
          q[i] = std::min(std::max(q[i], b.min_position_), b.max_position_);
        }
      }
    };

    const double step = 0.24;
    const double goal_thresh = 0.04;
    const double goal_bias = 0.25;
    const int    max_iters = 5000;
    const int    segment_checks = 10;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    auto dist = [&](const std::vector<double>& a, const std::vector<double>& b)
    {
      double s = 0.0;
      for (size_t i = 0; i < a.size(); ++i) { double d = a[i]-b[i]; s += d*d; }
      return std::sqrt(s);
    };

    auto collisionFreeState = [&](const std::vector<double>& q) -> bool
    {
      moveit::core::RobotState st(start_state);
      st.setJointGroupPositions(jmg, q);
      st.update();
      return !scene_->isStateColliding(st, getGroupName());
    };

    auto collisionFreeSegment = [&](const std::vector<double>& q1,
                                    const std::vector<double>& q2) -> bool
    {
      for (int i = 0; i <= segment_checks; ++i)
      {
        double a = static_cast<double>(i)/segment_checks;
        std::vector<double> q(q1.size());
        for (size_t j = 0; j < q.size(); ++j) q[j] = (1.0-a)*q1[j] + a*q2[j];
        if (!collisionFreeState(q)) return false;
      }
      return true;
    };

    clampToBounds(q_start);
    clampToBounds(q_goal);

    if (!collisionFreeState(q_start))
    {
      res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::START_STATE_IN_COLLISION;
      return false;
    }
    if (!collisionFreeState(q_goal))
    {
      res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::GOAL_IN_COLLISION;
      return false;
    }

    std::vector<Node> tree;
    tree.push_back(Node{q_start, -1});
    int goal_idx = -1;

    for (int it = 0; it < max_iters; ++it)
    {
      // TODO: Sample a random configuration q_rand
      // HINT: With probability goal_bias, sample the goal
      //       Otherwise, sample uniformly within joint bounds
      //       Use clampToBounds() to ensure the sample is valid
      std::vector<double> q_rand(q_start.size());
      if (uni01(rng) < goal_bias)
      {
        // Random sample should be goal configuration
        q_rand = q_goal;
      }
      else
      {
        for (size_t i = 0; i < q_rand.size() && i < var_count; ++i)
        {
          const auto& b = v_bounds[i];
          // Finish implementation of random sample
          if (b.position_bounded_)
          {
            std::uniform_real_distribution<double> unib(b.min_position_, b.max_position_);
            q_rand[i] = unib(rng);
          }
          else
          {
            q_rand[i] = 0.0; // Unbounded joint, set to zero or some default value
          }
        }
      }
      clampToBounds(q_rand);

      // TODO: Find the nearest node in the tree to q_rand
      // HINT: Use dist() to find closest node
      //       Store the index of the nearest node in variable 'nn'
      int nn = 0;
      double best = std::numeric_limits<double>::max();
      for (size_t i = 0; i < tree.size(); ++i)
      {
        double d = dist(tree[i].q, q_rand);
        if (d < best)
        {
          best = d;
          nn = i;
        }
      }

      // Steer from nearest node toward q_rand by at most distance 'step'
      std::vector<double> q_new(q_start.size());
      if (best <= step)
      {
        q_new = q_rand;
      }
      else
      {
        double s = step / best;
        for (size_t i = 0; i < q_new.size(); ++i)
          q_new[i] = tree[nn].q[i] + s*(q_rand[i] - tree[nn].q[i]);
      }
      clampToBounds(q_new);

      // TODO: Check if the path from tree[nn].q to q_new is collision-free
      // HINT: Use collisionFreeSegment(tree[nn].q, q_new)
      //       If in collision, skip to next iteration with 'continue'
      if (!collisionFreeSegment(tree[nn].q, q_new))
      {
        continue;
      }

      // TODO: Add q_new to the tree with parent nn
      // HINT: tree.push_back(Node{q_new, nn});
      //       Store the new node's index for the next check
      tree.push_back(Node{q_new, nn});
      int new_idx = tree.size() - 1;

      // TODO: Check if q_new is close enough to q_goal
      // HINT: If dist(q_new, q_goal) < goal_thresh AND
      //       collisionFreeSegment(q_new, q_goal) is collision-free,
      //       then add q_goal to tree and set goal_idx, then break
      if (dist(q_new, q_goal) < goal_thresh &&
          collisionFreeSegment(q_new, q_goal))
      {
        tree.push_back(Node{q_goal, new_idx});
        goal_idx = tree.size() - 1;
        break;
      }
      
    }

    if (goal_idx < 0)
    {
      res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
      return false;
    }

    // TODO: Reconstruct the path from start to goal
    // HINT: Start at goal_idx and follow parent pointers back to start
    //       Store configurations in 'path', then reverse it
    std::vector<std::vector<double>> path;
    for (int idx = goal_idx; idx != -1; idx = tree[idx].parent)
    {
      path.push_back(tree[idx].q);
    }
    std::reverse(path.begin(), path.end());
    

    moveit::core::RobotState st(start_state);
    robot_trajectory::RobotTrajectory traj(model_, jmg);
    const double dt = 0.1;
    bool first = true;
    for (const auto& q : path)
    {
      st.setJointGroupPositions(jmg, q);
      st.update();
      traj.addSuffixWayPoint(st, first ? 0.0 : dt);
      first = false;
    }

    res.trajectory_ = std::make_shared<robot_trajectory::RobotTrajectory>(traj);
    res.planning_time_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    return true;
  }

  bool solve(planning_interface::MotionPlanDetailedResponse& res) override
  {
    planning_interface::MotionPlanResponse simple;
    if (!solve(simple)) return false;
    res.trajectory_.push_back(simple.trajectory_);
    res.description_.push_back("plan");
    res.processing_time_.push_back(simple.planning_time_);
    res.error_code_ = simple.error_code_;
    return true;
  }

private:
  moveit::core::RobotModelConstPtr model_;
  planning_scene::PlanningSceneConstPtr scene_;
};

} // namespace rrt_planner
