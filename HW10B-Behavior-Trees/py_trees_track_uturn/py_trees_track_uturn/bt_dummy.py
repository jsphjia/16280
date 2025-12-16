#!/usr/bin/env python3
import py_trees
import time

# Blackboard keys (shared memory for BT nodes)
battery_level_key = "battery_level"
charging_key = "charging_mode"


# ===========================================================
#  Condition and Action Behaviors
# ===========================================================
class CheckBatteryOK(py_trees.behaviour.Behaviour):
    def __init__(self, threshold=30, name="CheckBatteryOK"):
        super().__init__(name)
        self.threshold = threshold

    def update(self):
        bb = py_trees.blackboard.Blackboard()   # The Blackboard is shared memory for the behavior tree.
                                                # All nodes can read/write values here.

        level = bb.get(battery_level_key)
        charging = bb.get(charging_key)

        # Do not allow navigation while charging
        if charging:
            return py_trees.common.Status.FAILURE

        if level > self.threshold:
            print(f"[Battery OK] level={level}%  ")
            return py_trees.common.Status.SUCCESS
        else:
            print(f"[Battery OK] level={level}%  ")
            return py_trees.common.Status.FAILURE


class CheckBatteryLow(py_trees.behaviour.Behaviour):
    def __init__(self, threshold=30, name="CheckBatteryLow"):
        super().__init__(name)
        self.threshold = threshold

    def update(self):
        bb = py_trees.blackboard.Blackboard()   # read shared state
        level = bb.get(battery_level_key)
        charging = bb.get(charging_key)

        # Keep charging branch active until full
        if charging:
            print(f"[Battery LOW/Charging] level={level}%  ")
            return py_trees.common.Status.SUCCESS

        if level <= self.threshold:
            print(f"[Battery LOW] level={level}%  ")
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.FAILURE


class FollowPath(py_trees.behaviour.Behaviour):
    # Action: navigation behavior
    def __init__(self):
        super().__init__("FollowPath")

    def update(self):
        print("[FollowPath] Navigating to goal...")
        return py_trees.common.Status.SUCCESS


class GoToDock(py_trees.behaviour.Behaviour):
    # Action: move toward docking station
    def __init__(self):
        super().__init__("GoToDock")

    def update(self):
        print("[GoToDock] Moving to docking station...")
        return py_trees.common.Status.RUNNING   # keeps running each tick


# ===========================================================
#  Build the BT structure
# ===========================================================
def create_root():

    # Sequence: run navigation only when battery OK
    # (Sequence = all children must succeed in order)
    nav_sequence = py_trees.composites.Sequence("Navigate", memory=False)
    nav_sequence.add_children([CheckBatteryOK(), FollowPath()])

    # Sequence: run charging behavior when battery low
    charge_sequence = py_trees.composites.Sequence("Charging", memory=False)
    charge_sequence.add_children([CheckBatteryLow(), GoToDock()])

    # Selector: chooses first SUCCESS/RUNNING child

    root = py_trees.composites.Selector("Root", memory=False)
    root.add_children([nav_sequence, charge_sequence])

    return root


# ===========================================================
#  Main Loop: simulate world + tick BT
# ===========================================================
if __name__ == "__main__":
    bb = py_trees.blackboard.Blackboard()
    bb.set(battery_level_key, 100)
    bb.set(charging_key, False)

    root = create_root()
    tree = py_trees.trees.BehaviourTree(root)

    # Generate tree diagram and save to the current folder
    py_trees.display.render_dot_tree(root, name="bt_tree", target_directory=".")

    print("\n--- Running Behavior Tree (Ctrl+C to stop) ---\n")

    battery = 100
    charging = False

    try:
        while True:
            # Simple battery model: drain while navigating, charge at dock
            if not charging:
                battery -= 5
            else:
                battery += 10

            battery = max(0, min(100, battery))
            bb.set(battery_level_key, battery)

            # Enter charging mode at 30% or below
            if battery <= 30:
                charging = True

            # Exit charging mode only after reaching 80%
            if battery >= 80 and charging:
                charging = False

            # Update Blackboard so BT conditions can react
            bb.set(charging_key, charging)

            # Tick: BT evaluates conditions and chooses branch
            tree.tick()
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStopped.")
