# Pose Synchronizer

This node synchronizes the LiDAR odometry (from HDL Graph SLAM) and the visual odometry (from ORB-SLAM3).

Since the LiDAR and camera are not supposed to by triggered at the exact same timestamps, we use the camera timestamps as reference, i.e.:
- Camera poses: no processing required
- LiDAR poses: interpolate according to image timestamps
- Camera images: no processing required
- LiDAR point clouds: move to image timestamp using LiDAR odometry


### Usage

- Start roscore: `roscore`
- Visualize in rviz: `roscd pose_synchronizer`; `rviz -d rviz/combined.rviz`
- Run CMRNext: `rosrun cmrnext cmrnext_ros_node.py`
- Run optimizer: `rosrun optimization_utils optimization_utils`
- Run synchronizer: `roslaunch pose_synchronizer pose_synchronizer.launch`
- _Wait until the vocabulary has been loaded_
- Play bagfile: `roslaunch pose_synchronizer play_bag.launch`
