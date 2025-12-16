# Autonomous TurtleBot

This project implements a vision-based steering angle prediction model for autonomous navigation using the TurtleBot platform. You will collect data via teleoperation, preprocess camera images, and train a PyTorch-based regression model to predict steering commands from RGB images. The model learns to predict the angular velocity (ω) command directly from RGB images captured by the front camera of the TurtleBot.

---

## Package Structure

The workspace contains two main components: a `model_generation` folder for data processing and training, and a `self_driving` ROS2 package for deployment.

### model_generation Directory

```
model_generation/
├── ckpt_best.pt                    # Trained model weights
├── data/
│   ├── processed/
│   │   └── merged_dataset/
│   │       ├── images/             # Extracted camera frames
│   │       ├── index_smooth.json   # Smoothed steering commands
│   │       └── index_split.json    # Train/val/test split labels
│   └── raw/                        # ROS2 bag files
├── dataset.py                      # Dataset loader class
├── eval.py                         # Model evaluation script
├── model.py                        # Neural network architecture
├── runs/                           # TensorBoard logs
├── train.ipynb                     # Training notebook (Google Colab)
└── util/
    ├── data_split.py               # Train/val/test splitting
    ├── extract_ros_bag.py          # ROS bag extraction
    ├── merge_dataset.py            # Dataset merging utility
    ├── plot_data.py                # Data visualization
    └── smooth_omega.py             # Temporal smoothing
```

### self_driving ROS Package

```
self_driving/
├── launch/                         # ROS2 launch files
├── resource/                       # Workflow videos and reference data
├── self_driving/
│   ├── __init__.py
│   ├── model.py                    # Neural network (copy from model_generation)
│   └── steering_inference_image.py # ROS2 inference node
├── package.xml
├── setup.cfg
└── setup.py
```

---

## Installation

Install the required dependencies in the following order:

```bash
sudo apt update
sudo apt install python3-pip
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install tensorboard
pip install tqdm
sudo apt install ros-humble-cv-bridge
```

**Note:** Model training is performed using Google Colab with GPU acceleration. You will upload the `train.ipynb` file to your Drive and follow the instructions within the notebook to link your dataset.

---

## Configuration

### Teleoperation Script Modifications

The default teleoperation script publishes velocity commands at 10 Hz, which provides insufficient temporal resolution for our regression task. We need to increase the publishing rate to 33 Hz and decrease the angular velocity step size for finer control.

On the robot computer, open the teleoperation script:

```bash
nano turtlebot3_ws/src/turtlebot3/turtlebot3_teleop/turtlebot3_teleop/script/teleop_keyboard.py
```

**Modify line 89** to increase the command velocity frequency:

```python
rlist, _, _ = select.select([sys.stdin], [], [], 0.03)  # 33 Hz (originally 0.1s for 10 Hz)
```

**Modify line 60** to decrease the angular velocity step size:

```python
ANG_VEL_STEP_SIZE = 0.05  # Finer resolution for smoother steering
```

After making these changes, compile and source your workspace:

```bash
colcon build
source install/setup.bash
```

---

## Data Collection

### Recording Procedure

Data collection is performed by manually teleoperating the TurtleBot along the track while recording synchronized camera images and velocity commands. Place the TurtleBot on the track, centered on the blue lines.

**Data Guidelines:**
- The blue track must be clearly visible in every frame
- Minor occlusions at corners are acceptable
- Most of the track should remain within the camera's field of view at all times
- Refer to the "good data" reference video in `self_driving/resource/` for examples

### Launch Sequence

On the robot computer, execute the following commands:

```bash
ros2 launch turtlebot3_bringup robot.launch.py

ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[320, 240]" \
    -p qos_overrides.image_raw.publisher.reliability:=best_effort \
    -p qos_overrides.image_raw.publisher.history:=keep_last \
    -p qos_overrides.image_raw.publisher.depth:=10 \
    -p qos_overrides.image_raw.publisher.durability:=volatile

ros2 run turtlebot3_teleop teleop_keyboard

v4l2-ctl -d /dev/video0 -p 5  # Set camera frame rate to 5 Hz
```

### Recording Commands

Navigate to `model_generation/data/raw/` and record your data:

```bash
ros2 bag record /image_raw/compressed /cmd_vel
```

You must record two separate bag files:
- **Bag 1:** At least two complete laps counter-clockwise (CCW)
- **Bag 2:** At least two complete laps clockwise (CW)

The bidirectional data ensures the model learns robust steering behavior in both directions.

---

## Dataset Processing

Use the provided utility scripts to process your raw ROS bag files. Replace `rosbag2_xxx` and `rosbag2_yyy` with your actual bag file names throughout.

### Extract Images and Commands

Each ROS bag is processed to extract camera frames from `/image_raw/compressed` and velocity commands from `/cmd_vel`. For each camera frame, the script retrieves the closest velocity command message based on timestamp alignment.

```bash
cd model_generation

python3 util/extract_ros_bag.py --bag data/raw/rosbag2_xxx \
    --out data/processed/bag_1 \
    --image-topic /image_raw/compressed \
    --cmd-topic /cmd_vel

python3 util/extract_ros_bag.py --bag data/raw/rosbag2_yyy \
    --out data/processed/bag_2 \
    --image-topic /image_raw/compressed \
    --cmd-topic /cmd_vel
```

The output consists of decoded RGB images in PNG format and a JSON file containing the paired velocity commands.

### Split into Train/Validation/Test Sets

The extracted data is split into non-overlapping subsets for training (70%), validation (20%), and testing (10%). The fixed random seed ensures reproducibility.

```bash
python3 util/data_split.py --index data/processed/bag_1/index.json --seed 123
python3 util/data_split.py --index data/processed/bag_2/index.json --seed 123
```

### Merge Datasets

Combine both CW and CCW datasets into a unified training set:

```bash
python3 util/merge_dataset.py --runs data/processed/bag_1 data/processed/bag_2 \
    --out data/processed/merged_dataset --copy
```

### Apply Temporal Smoothing

The recorded velocity commands are discrete with a step size of 0.05 rad/s. Training a regressor directly on discrete data can result in jerky predictions. Temporal smoothing creates a more continuous label distribution.

```bash
python3 util/smooth_omega.py --index data/processed/merged_dataset/index_split.json
```

The smoothed values replace the original labels and are saved as `index_smooth.json`.

### Visualize Dataset

To inspect the command distribution and verify data integrity:

```bash
python3 util/plot_data.py data/processed/merged_dataset/index_split.json
python3 util/plot_data.py data/processed/merged_dataset/index_smooth.json
```

This generates histograms of angular velocity values and allows you to compare the effect of smoothing.

---

## Training

The model architecture consists of a pretrained ResNet-18 backbone for feature extraction, followed by a lightweight MLP regression head. The network is trained using mean absolute error (MAE) loss to minimize steering prediction error.

### Google Colab Setup

1. Navigate to `model_generation/train.ipynb`
2. Upload the notebook to Google Colab
3. Enable GPU acceleration in Runtime settings
4. Follow the in-notebook instructions to:
   - Mount your Google Drive
   - Upload your processed dataset
   - Configure hyperparameters (learning rate, batch size, epochs)
   - Apply data augmentation (TopCrop, brightness, contrast, hue, saturation)
   - Monitor training progress

The best model checkpoint (`ckpt_best.pt`) will be automatically saved to your Drive upon training completion.

### TensorBoard Monitoring

To visualize training curves locally:

```bash
cd model_generation
python3 -m tensorboard.main --logdir runs
```

Access TensorBoard by navigating to `http://localhost:6006` in your browser.

---

## Evaluation

After training, evaluate the model's performance on the held-out test set:

```bash
cd model_generation

python3 eval.py --index data/processed/merged_dataset/index_smooth.json \
    --root data/processed/merged_dataset \
    --ckpt ckpt_best.pt \
    --split test \
    --outdir eval_out \
    --save-overlays \
    --flip-sign \
    --short-side 224 \
    --top-crop 0.2
```

This script computes performance metrics (MAE and RMSE) and generates visualization outputs in the `eval_out/` folder, including prediction overlays on test images.

---

## Deployment

### Robot Setup

Place the TurtleBot at the center of the track, facing the forward driving direction. Ensure the camera is positioned identically to your data collection setup.

### Launch Sequence

**On the robot computer:**

```bash
ros2 launch turtlebot3_bringup robot.launch.py

ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[320, 240]" \
    -p qos_overrides.image_raw.publisher.reliability:=best_effort \
    -p qos_overrides.image_raw.publisher.history:=keep_last \
    -p qos_overrides.image_raw.publisher.depth:=10 \
    -p qos_overrides.image_raw.publisher.durability:=volatile

v4l2-ctl -d /dev/video0 -p 5  # Maintain 5 Hz camera frame rate
```

**On the host computer:**

Monitor the system topics and launch the inference node:

```bash
# Verify camera frame rate
ros2 topic hz /image_raw/compressed
ros2 run rqt_image_view rqt_image_view image:=/image_raw/compressed

# Monitor command velocity output
ros2 topic hz /cmd_vel
ros2 topic echo /cmd_vel

# Launch inference node
ros2 launch self_driving steering_inference.launch.xml
```

Ensure the `/image_raw/compressed` frame rate remains stable at 5 Hz. Frame drops will introduce latency in the inference loop, resulting in jerky steering and unstable control.

---

## Expected Behavior

Upon successful deployment, the robot should autonomously follow the blue track using only camera input. The steering commands should be smooth and responsive, with the robot maintaining a centered position within the lane boundaries. In our testing, the robot was able to navigate the path in the arena 10+ times without error before it was stopped. 

---

## Troubleshooting

**Unstable Frame Rate:**
- Verify that no additional nodes are subscribing to `/image_raw/compressed`
- Confirm camera configuration using `v4l2-ctl -d /dev/video0 -p 5`
- Close `rqt_image_view` during autonomous operation to reduce computational load

**Poor Steering Performance:**
- Ensure camera position and orientation match training data collection setup
- Verify that you are using the smoothed dataset (`index_smooth.json`) for training
- Check that environmental lighting conditions are similar to training conditions
- Confirm that all preprocessing parameters (crop, resize, normalization) match training configuration

**Model Loading Errors:**
- Verify the file path to `ckpt_best.pt` is correct in the launch file
- Ensure `model.py` in `self_driving/` is identical to the version used during training
- Confirm that PyTorch versions match between training and deployment environments

