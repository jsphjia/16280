#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import py_trees
import py_trees.common
import py_trees.behaviour
import py_trees.trees


# ============================================================
#  SYSTEM DONE SEQUENCE
# ============================================================

class SystemDoneCondition(py_trees.behaviour.Behaviour):
    """Checks if mission/system is done."""
    def __init__(self, name, node: Node):
        super().__init__(name=name)
        self.node = node

    def update(self):
        flag = self.node.get_parameter("system_done").value
        if flag:
            self.node.get_logger().info("[SystemDoneCondition] system_done=True")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class IdleAction(py_trees.behaviour.Behaviour):
    """Put robot into idle mode."""
    def __init__(self, name, node: Node):
        super().__init__(name=name)
        self.node = node

    def update(self):
        self.node.get_logger().info("[IdleAction] Entering IDLE mode")
        self.node.set_mode("idle")
        return py_trees.common.Status.SUCCESS


# ============================================================
#  UTURN SEQUENCE
# ============================================================

class UTurnCondition(py_trees.behaviour.Behaviour):
    """Checks whether U-turn is requested by operator."""
    def __init__(self, name, node: Node):
        super().__init__(name=name)
        self.node = node

    def update(self):
        need_uturn = self.node.get_parameter("u_turn_needed").value
        if need_uturn:
            self.node.get_logger().info("[UTurnCondition] u_turn_needed=True")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE


class UTurnAction(py_trees.behaviour.Behaviour):
    """Perform U-turn in place (steering node handles twist)."""
    def __init__(self, name, node: Node):
        super().__init__(name=name)
        self.node = node

    def update(self):
        self.node.set_mode("uturn")
        self.node.get_logger().info("[UTurnAction] Executing U-turn...")
        return py_trees.common.Status.RUNNING


# ============================================================
#  TRACK ACTION
# ============================================================

class TrackAction(py_trees.behaviour.Behaviour):
    """Default track-following behavior."""
    def __init__(self, name, node: Node):
        super().__init__(name=name)
        self.node = node

    def update(self):
        self.node.set_mode("track")
        self.node.get_logger().info("[TrackAction] Tracking path...")
        return py_trees.common.Status.SUCCESS


# ============================================================
#  BT NODE WRAPPER
# ============================================================

class BehaviorTreeNode(Node):
    def __init__(self):
        super().__init__("track_uturn_bt")

        # parameters
        self.declare_parameter("system_done", False)
        self.declare_parameter("u_turn_needed", False)

        # Mode publisher (track, uturn, idle)
        self.mode_pub = self.create_publisher(String, "/behavior_mode", 10)
        self.current_mode = None

        # Build behavior tree
        root = self.create_tree()
        self.tree = py_trees.trees.BehaviourTree(root)

        # Tick tree at 2 Hz
        self.timer = self.create_timer(0.5, self.tick_tree)

        self.get_logger().info("Behavior Tree (SystemDone + UTurn + Track) started.")

    # -------------------------------
    # mode publisher
    # -------------------------------
    def set_mode(self, mode: str):
        if mode == self.current_mode:
            return
        self.current_mode = mode
        # ================ TBD ================
        # complete the publisher node that publishes String type message for the 'mode' variable
        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)
        #================ TBD end ================
        self.get_logger().info(f"[BT] behavior_mode = {mode}")

    # -------------------------------
    # build BT
    # -------------------------------
    def create_tree(self):
        # under each node, based on the given behavior tree diagram and the
        # py_trees examples from the dummy python script and ros2 node,
        # fill in the TBD section (...) below to include
        # condition, action, node type (sequence or selector), and the children nodes
        
        # ================  TBD ================
        
        # -------- SystemDone  --------
        sys_cond = SystemDoneCondition("SystemDoneCondition", self)
        idle_act = IdleAction("IdleAction", self)
        sys_seq = py_trees.composites.Sequence("SystemDoneSeq", memory=False)
        sys_seq.add_children([sys_cond, idle_act])

        # -------- UTurn  --------
        uturn_cond = UTurnCondition("UTurnCondition", self)
        uturn_act  = UTurnAction("UTurnAction", self)
        uturn_seq  = py_trees.composites.Sequence("UTurnSeq", memory=False)
        uturn_seq.add_children([uturn_cond, uturn_act])

        # -------- Behavior  (UTurn vs Track) --------
        track_act = TrackAction("TrackAction", self)
        behavior_selector = py_trees.composites.Selector("BehaviorSelector", memory=False)
        behavior_selector.add_children([uturn_seq, track_act])

        # -------- Top (SystemDone vs Behavior) --------
        root = py_trees.composites.Selector("RootSelector", memory=False)
        root.add_children([sys_seq, behavior_selector])

        #================ TBD END ================
        py_trees.display.render_dot_tree(root, name="bt_tree", target_directory=".")

        return root

    # -------------------------------
    # tick
    # -------------------------------
    def tick_tree(self):
        self.tree.tick()


# ============================================================
#  MAIN
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = BehaviorTreeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
