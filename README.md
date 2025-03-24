# MDPCalib
[**arXiv**](https://arxiv.org/abs/2404.17298) | [**IEEE Xplore**](https://ieeexplore.ieee.org/document/10694691) | [**Website**](https://calibration.cs.uni-freiburg.de/) | [**Video**](https://youtu.be/L1MwAenzd6g)

This repository is the official implementation of the paper:

> **Automatic Target-Less Camera-LiDAR Calibration from Motion and Deep Point Correspondences**
>
> [Kürsat Petek](http://www2.informatik.uni-freiburg.de/~petek/)&ast;, [Niclas Vödisch](https://vniclas.github.io/)&ast;, [Johannes Meyer](http://www2.informatik.uni-freiburg.de/~meyerjo/), [Daniele Cattaneo](https://rl.uni-freiburg.de/people/cattaneo), [Abhinav Valada](https://rl.uni-freiburg.de/people/valada), and [Wolfram Burgard](https://www.utn.de/person/wolfram-burgard/). <br>
> &ast;Equal contribution. <br>
>
> *IEEE Robotics and Automation Letters*, vol. 9, issue 11, pp. 9978-9985, November 2024.

<p align="center">
  <img src="./assets/mdpcalib_overview.png" alt="Overview of MDPCalib approach" width="800" />
</p>

If you find our work useful, please consider citing our paper:
```
@article{petek2024mdpcalib,
  author={Petek, Kürsat and Vödisch, Niclas and Meyer, Johannes and Cattaneo, Daniele and Valada, Abhinav and Burgard, Wolfram},
  journal={IEEE Robotics and Automation Letters},
  title={Automatic Target-Less Camera-LiDAR Calibration From Motion and Deep Point Correspondences},
  year={2024},
  volume={9},
  number={11},
  pages={9978-9985}
}
```


## 📔 Abstract

Sensor setups of robotic platforms commonly include both camera and LiDAR as they provide complementary information. However, fusing these two modalities typically requires a highly accurate calibration between them. In this paper, we propose MDPCalib which is a novel method for camera-LiDAR calibration that requires neither human supervision nor any specific target objects. Instead, we utilize sensor motion estimates from visual and LiDAR odometry as well as deep learning-based 2D-pixel-to-3D-point correspondences that are obtained without in-domain retraining. We represent the camera-LiDAR calibration as a graph optimization problem and minimize the costs induced by constraints from sensor motion and point correspondences. In extensive experiments, we demonstrate that our approach yields highly accurate extrinsic calibration parameters and is robust to random initialization. Additionally, our approach generalizes to a wide range of sensor setups, which we demonstrate by employing it on various robotic platforms including a self-driving perception car, a quadruped robot, and a UAV.


## 👩‍💻 Code

### 💻 Development

#### Docker 🐋

Tested with `Docker version 28.0.1` and `Docker Compose version v2.33.1`.

- To build the image, run `docker compose build` in the root of this repository.
- Prepare using GUIs in the container: `xhost +local:docker`.
- Start container and mount rosbags: `docker compose run -v PATH_TO_DATA:/data -it mdpcalib`
- Connect to a running container: `docker compose exec -it mdpcalib bash`


#### Githooks ✅

We used multiple githooks during the development of this code. You can set up them with the following steps:

1. Create a venv or conda environment. Make sure to source that before committing.
2. Install requirements: `pip install -r requirements.txt`
3. Install CMRNext requirements for pylint to work: `pip install -r src/CMRNext/rquirements.txt`
4. Install [pre-commit](https://pre-commit.com/) githook scripts: `pre-commit install`

Python formatter ([yapf](https://github.com/google/yapf), [iSort](https://github.com/PyCQA/isort)) settings can be set in [pyproject.toml](pyproject.toml). C++ formatter ([ClangFormat](https://clang.llvm.org/docs/ClangFormat.html)) settings are in [.clang-format](.clang-format).

This will automatically run the pre-commit hooks on every commit. You can skip this using the `--no-verify` flag, e.g., `commit -m "Update node" --no-verify`.
To run the githooks on all files independent of doing a commit, use `pre-commit run --all-files`


### 💾 Data

In the public release of our MDPCalib, we provide instructions for running camera-LiDAR calibration on the KITTI dataset.

#### Generating a KITTI rosbag 🐈

> &#x26a0;&#xfe0f; **Note:** If you created the rosbag before commit [19645fb](https://github.com/robot-learning-freiburg/MDPCalib/tree/19645fb5b224d45b6a56c6daa94ed873e911f10c) (Mar 13, 2025), please re-create it following the updated instructions.

- Install the provided `kitti2bag` package from within the package directory: `pip install -e .`
- Download the raw "synced+rectified" and "calibration" data for an odometry sequence. The mapping is available [here](https://github.com/tomas789/kitti2bag/issues/10#issuecomment-352962278). In the following, we will assume that the files will be downloaded to `/data/kitti/`.
- E.g., for sequence 00, download the residential data `2011_10_03_drive_0027`:
    - https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0027/2011_10_03_drive_0027_sync.zip
    - https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_calib.zip
- Unzip both files.
    - `unzip 2011_10_03_calib.zip`
    - `unzip 2011_10_03_drive_0027/2011_10_03_drive_0027_sync.zip`
- Now, we replace the raw velodyne data with the point clouds from the odometry benchmark as these have been motion compensated. Download the "velodyne laser data":
    - https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip
- It is sufficient to unzip the same sequence as downloaded above, e.g.,
    - `rm -rf 2011_10_03/2011_10_03_drive_0027_sync/velodyne_points/data`
    - `unzip -j -d 2011_10_03/2011_10_03_drive_0027_sync/velodyne_points/data data_odometry_velodyne.zip 'dataset/sequences/00/velodyne/*.bin'`
- Then run: `kitti2bag -t 2011_10_03 -r 0027 raw_synced`.
- This will result in a rosbag: `kitti_2011_10_03_drive_0027_synced.bag`.
- For the following instructions, we will assume that the rosbag is located at `/data/kitti/`.
    - The folder can be changed in the launchers [play_bag_kitti_left.launch](src/pose_synchronizer/launch/play_bag_kitti_left.launch) and [play_bag_kitti_right.launch](src/pose_synchronizer/launch/play_bag_kitti_right.launch).

#### Downloading model weights 🏋️

Please download the model weights of CMRNext from this link and store them under: `/data/cmrnext`.
- Model weights: https://calibration.cs.uni-freiburg.de/downloads/cmrnext_weights.zip

### 🏃 Running the calibration

For changing the configuration settings, please consult the [config file](src/calib_cfg/config/config.yaml).
Note that there we also specify the name of the experiment that creates an output folder under `/root/catkin_ws/src/mdpcalib/experiments`.
To prevent overwriting previous results, names of experiments must be unique.

Please execute the following steps in separate terminals:

1. Start a roscore: `roscore`
2. [Optional] Start rviz: `roscd pose_synchronizer; rviz -d rviz/combined.rviz`
3. Launch CMRNext: `roslaunch cmrnext cmrnext_kitti.launch`
4. Launch the optimizer: `roslaunch optimization_utils optimizer.launch`
5. Launch the data synchronizer: `roslaunch pose_synchronizer pose_synchronizer_fastlo_kitti.launch`
6. Wait until the ORB vocabulary has been loaded.
7. Play the data (left camera): `roslaunch pose_synchronizer play_bag_kitti_left.launch`
    - Alternatively (right camera): `roslaunch pose_synchronizer play_bag_kitti_right.launch`
8. [Optional] Once the fine calibration is finished, you could stop playing the data (Ctrl + c).


## 👩‍⚖️  License

For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. Components of other works are released under their original license.
For any commercial purpose, please contact the authors.


## 🙏 Acknowledgment

In our work and experiments, we have used components from many other works. We thank the authors for open-sourcing their code. In no specific order, we list source repositories:
- CMRNext: https://github.com/robot-learning-freiburg/CMRNext
- ORB SLAM3 ROS Wrapper: https://github.com/thien94/orb_slam3_ros_wrapper
- kitti2bag: https://github.com/tomas789/kitti2bag
- FAST-LO: https://github.com/hku-mars/LiDAR_IMU_Init


This work was funded by the German Research Foundation (DFG) Emmy Noether Program grant No 468878300 and an academic grant from NVIDIA.
<br><br>
<p float="left">
  <a href="https://www.dfg.de/en/research_funding/programmes/individual/emmy_noether/index.html"><img src="./assets/dfg_logo.png" alt="DFG logo" height="100"/></a>
</p>
