#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node

import py_trees

# Blackboard keys (shared memory for BT nodes)
battery_level_key = "battery_level"
charging_key = "charging_mode"


# ===========================================================
#  Behaviours: Conditions and Actions
# ===========================================================
class CheckBatteryOK(py_trees.behaviour.Behaviour):
    # Condition: battery must be above threshold AND not in charging mode
    def __init__(self, threshold=30.0, name="CheckBatteryOK"):
        super().__init__(name)
        self.threshold = threshold

    def update(self):
        bb = py_trees.blackboard.Blackboard()
        level = bb.get(battery_level_key)
        charging = bb.get(charging_key)

        if level is None:
            return py_trees.common.Status.FAILURE

        # While charging_mode is True, suppress the navigation branch
        if charging:
            print(f"[Battery OK?] level={level:.1f}% but CHARGING MODE FAILURE")
            return py_trees.common.Status.FAILURE

        if level > self.threshold:
            print(f"[Battery OK] level={level:.1f}% ")
            return py_trees.common.Status.SUCCESS
        else:
            print(f"[Battery OK] level={level:.1f}% ")
            return py_trees.common.Status.FAILURE


class CheckBatteryLow(py_trees.behaviour.Behaviour):
    # Condition: succeeds when battery low OR robot is in charging mode
    def __init__(self, threshold=30.0, name="CheckBatteryLow"):
        super().__init__(name)
        self.threshold = threshold

    def update(self):
        bb = py_trees.blackboard.Blackboard()
        level = bb.get(battery_level_key)
        charging = bb.get(charging_key)

        if level is None:
            return py_trees.common.Status.FAILURE

        # Once charging_mode is active, keep the Charging branch selected
        if charging:
            print(f"[Battery LOW/Charging] level={level:.1f}% ")
            return py_trees.common.Status.SUCCESS

        if level <= self.threshold:
            print(f"[Battery LOW] level={level:.1f}% ")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class FollowPath(py_trees.behaviour.Behaviour):
    # Action: navigation behavior
    def __init__(self):
        super().__init__("FollowPath")

    def update(self):
        print("[FollowPath] Navigating...")
        return py_trees.common.Status.SUCCESS


class GoToDock(py_trees.behaviour.Behaviour):
    # Action: docking/charging behavior
    def __init__(self):
        super().__init__("GoToDock")

    def update(self):
        print("[GoToDock] Docking (charging mode)…")
        return py_trees.common.Status.RUNNING  # keep running while charging


# ===========================================================
#  Tree Construction
# ===========================================================
def create_root():
    # Sequence: Navigation branch (requires battery OK, then follow path)
    nav_sequence = py_trees.composites.Sequence("Navigate", memory=False)
    nav_sequence.add_children([CheckBatteryOK(), FollowPath()])

    # Sequence: Charging branch (requires battery low/charging, then dock)
    charge_sequence = py_trees.composites.Sequence("Charging", memory=False)
    charge_sequence.add_children([CheckBatteryLow(), GoToDock()])

    # Selector: tries Navigate first; if it fails, runs Charging
    root = py_trees.composites.Selector("Root", memory=False)
    root.add_children([nav_sequence, charge_sequence])
    return root


# ===========================================================
#  ROS 2 Node
# ===========================================================
class BatteryBTNode(Node):
    def __init__(self):
        super().__init__("battery_bt_node")

        # Declare parameter $ros2 param set /battery_bt_node initial_battery_level 50 to set initial value
        self.declare_parameter("initial_battery_level")

        self.bt_initialized = False
        self.battery = None
        self.charging = False

        # Timer: wait until parameter is provided
        self.startup_timer = self.create_timer(0.5, self.wait_for_param)

    def wait_for_param(self):
        param = self.get_parameter("initial_battery_level")

        if param.type_ != param.Type.NOT_SET:
            # Parameter is available initialize battery
            self.battery = float(param.value)
            self.get_logger().info(f"Received initial_battery_level={self.battery}")

            # Initialize blackboard
            self.bb = py_trees.blackboard.Blackboard()
            self.bb.set(battery_level_key, self.battery)
            self.bb.set(charging_key, self.charging)

            # Build BT
            self.root = create_root()
            self.tree = py_trees.trees.BehaviourTree(self.root)

            # Start ticking at 1 Hz
            self.timer = self.create_timer(1.0, self.tick_tree)

            # Stop startup timer
            self.startup_timer.cancel()
            self.bt_initialized = True

        else:
            self.get_logger().info("Waiting for parameter: initial_battery_level...")

    def tick_tree(self):
        if not self.bt_initialized:
            return

        # Simple battery model for demo: drain while not charging, charge otherwise
        if not self.charging:
            self.battery -= 5.0
        else:
            self.battery += 10.0

        # Clamp to [0, 100]
        self.battery = max(0.0, min(100.0, self.battery))

        # Enter charging mode when battery is low (≤ 30%)
        if self.battery <= 30.0:
            self.charging = True

        # Leave charging mode only after reaching 80%
        if self.battery >= 80.0 and self.charging:
            self.charging = False

        # Update Blackboard for BT conditions (battery + charging_mode)
        self.bb.set(battery_level_key, self.battery)
        self.bb.set(charging_key, self.charging)

        self.get_logger().info(
            f"[Node] battery={self.battery:.1f}% charging={self.charging}"
        )

        # One BT tick
        self.tree.tick()


def main(args=None):
    rclpy.init(args=args)
    node = BatteryBTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
