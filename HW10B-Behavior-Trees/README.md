# Decision Making Using Behaviour Tree 
This project will implement a decision making framework using Behaviour Trees. The Behaviour Trees implementation is done using pytrees, You will finally integrate it inside a ROS node and run it on your turtlebot. 

## Software package

The workspace contains a directory py_trees_track_uturn and a python script steering_inference_image_bt.py.

## py_trees_track_uturn directory:
The py_trees_track_uturn directory serves as the main repository for the behavior-tree implementation. It contains two example behavior trees implemented using py_trees: 
- bt_dummy.py - A python script with BT <br>
- bt_dummy_node.py - BT inside a ROS node <br>

├── test <br>
├── resource &nbsp;&nbsp;&nbsp; # contains workflow videos<br>
└── py_trees_track_uturn <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──_init.py	<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── bt_dummy.py &nbsp;&nbsp;&nbsp;# example python script for you to study <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── bt_dummy_node.py &nbsp;&nbsp;&nbsp;# example ROS node with BT for you to study <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── bt_node.py &nbsp;&nbsp;&nbsp; # final BT node you will implement<br>
├── package.xml <br>
├── setup.config <br>
└── setup.py <br>
 
The steering_inference_image_bt.py file should be added to your previous autonomous-bot repository. This new module determines the robot’s behavior based on the logic defined in the behavior-tree node. For additional details, please refer to the homework PDF and the workflow video.      

## Software dependencies to be installed:

```shell
$ sudo apt update
$ sudo apt upgrade
$ sudo apt install ros-humble-py-trees*
```

## Files to Modify 

You will have to update bt_node.py based on the instructions provided. Also modify steering_inference_image_bt.py accordingly. Add steering_inference_image_bt.py to HW10A workspace in the self_driving package. Modify the setup.py and the launhc file to include the new node.







## ROS2 Nodes

**Robot:**
- Robot bring up
```shell
$ ros2 launch turtlebot3_bringup robot.launch.py
$ v4l2-ctl -d /dev/video0 -p 5
$ ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[320, 240]" -p
qos_overrides.image_raw.publisher.reliability:=best_effort -p qos_overrides.
image_raw.publisher.history:=keep_last -p qos_overrides.image_raw.publisher.
depth:=10 -p qos_overrides.image_raw.publisher.durability:=volatile
```	
**Host Computer:**
- Decision making nodes
```shell
$ ros2 launch self_driving steering_inference.launch.xml
$ ros2 run py_trees_track_uturn bt_node
```	



