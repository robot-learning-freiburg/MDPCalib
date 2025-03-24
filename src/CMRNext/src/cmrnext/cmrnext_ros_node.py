import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import message_filters
import numpy as np
import ros_numpy
import rosbag
import rospkg
import rospy
import torch
import yaml
from cmrnext.cmrnext import CMRNext
from cmrnext.utils import average_quaternions, quat2mat, quaternion_from_matrix
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import MultiArrayDimension
from tqdm import tqdm

from calib_msgs.msg import (
    ImagePclCorrespondencesStamped,
    StringStamped,
    UInt16MultiArrayStamped,
)


class CMRNextNode:

    def __init__(self, model_weights: List[str], target_shape: Tuple[int, int],
                 downsample: bool) -> None:
        self.cv_bridge = CvBridge()
        self.rng = np.random.default_rng()

        self.model: CMRNext = None
        self.model_weights = model_weights
        self.target_shape = target_shape
        self.downsample = downsample
        self.initial_transform = None
        # Contains seq number of first and last pair used to compute the initial transform
        self.initial_transform_meta = []
        # Compute correspondences for every Nth pair
        self.pair_processing_frequency = 1
        # The optimizer only expects correspondences for N pairs
        self.number_image_pcl_pairs = np.inf
        # Percentage of correspondences (points) sent by the publisher
        self.amount_correspondences = 100  # Valid numbers in [0, 100]
        # Filter correspondences based on uncertainty.
        self.use_uncertainty = False
        # Img-Pcl pairs with an estimated yaw above this threshold are not processed. In degree.
        self.rotation_threshold = np.inf
        self.final_calibs = []

        self.synced_data_filename_subscriber: message_filters.Subscriber = None
        self.synced_data_filename_buffer = []
        self.msg_counter = 0
        self.prev_yaw = 0

        self.initial_transform_meta_subscriber: message_filters.Subscriber = None
        self.initial_transform_data_subscriber: message_filters.Subscriber = None

        self.correspondences_publisher: rospy.Publisher = None

        for weight in model_weights:
            if not Path(weight).exists():
                rospy.logerr(f"Model weights {weight} do not exist.")
                rospy.signal_shutdown("Model weights do not exist.")

    def _initialize_model(self, camera_info_msg: CameraInfo) -> None:
        intrinsic_calibration = np.array(camera_info_msg.P).reshape(3, 4)[:3, :3]
        if intrinsic_calibration.sum() < 10:
            intrinsic_calibration = np.array(camera_info_msg.K).reshape(3, 3)
        rospy.loginfo("Received intrinsic calibration of camera.")

        self.model = CMRNext(self.model_weights, [
            intrinsic_calibration[0, 0], intrinsic_calibration[1, 1], intrinsic_calibration[0, 2],
            intrinsic_calibration[1, 2]
        ], self.target_shape, self.downsample)
        rospy.loginfo("Initialized CMRNext model.")

        # If the initial transform has already been received, process the buffer
        if self.initial_transform is not None and len(self.initial_transform_meta) == 2:
            self._process_buffer()

    def pointcloud2_to_numpy(self, pcl_message: PointCloud2) -> ArrayLike:
        dtype_list = ros_numpy.point_cloud2.fields_to_dtype(pcl_message.fields,
                                                            pcl_message.point_step)
        cloud_arr = np.frombuffer(pcl_message.data, dtype_list)
        xyz_arr = np.zeros((cloud_arr["x"].shape[0], 3), dtype=float)
        xyz_arr[:, 0] = cloud_arr["x"].astype(float).copy()
        xyz_arr[:, 1] = cloud_arr["y"].astype(float).copy()
        xyz_arr[:, 2] = cloud_arr["z"].astype(float).copy()
        return xyz_arr.copy()

    def _cache_img_pcl_from_rosbag(self, synced_data_filename_msg: StringStamped) -> None:
        filename = Path(synced_data_filename_msg.data)
        while not filename.exists():
            time.sleep(.1)

        # Data is from before the optimization period, thus ignore and delete file.
        if len(self.initial_transform_meta
               ) > 0 and synced_data_filename_msg.header.seq < self.initial_transform_meta[0]:
            rospy.loginfo(f"Ignoring image and pcl with seq {synced_data_filename_msg.header.seq}")
            filename.unlink()
            return
        # Data is from after the optimization period, thus ignore, delete file, and
        # stop listening (then the publisher will also stop saving data).
        if len(self.initial_transform_meta
               ) == 2 and synced_data_filename_msg.header.seq > self.initial_transform_meta[1]:
            rospy.loginfo(
                f"Ignoring image and pcl with seq {synced_data_filename_msg.header.seq} and " +
                "stop filling buffer.")
            filename.unlink()
            # No need to cache any more data
            self.synced_data_filename_subscriber.unregister()
            return

        lidar_pose_msg = None
        with rosbag.Bag(filename, "r") as bag:
            for _, msg, _ in bag.read_messages(topics=["lidar_pose"]):
                lidar_pose_msg = msg

        # If the estimated rotation from LiDAR odometry is above a threshold, ignore this pair
        #  b/c the proiection will not be sufficiently accurate due to imperfect undistortion.
        quaternion = lidar_pose_msg.pose.orientation
        rotation = torch.from_numpy(
            R.from_quat([quaternion.x, quaternion.y, quaternion.z,
                         quaternion.w]).as_euler("yxz", degrees=True))
        yaw = rotation[2]
        relative_yaw = np.abs(yaw - self.prev_yaw)
        self.prev_yaw = yaw
        if relative_yaw > self.rotation_threshold:
            rospy.loginfo(
                f"Ignoring image and pcl with seq {synced_data_filename_msg.header.seq} due to "
                f"large yaw ({relative_yaw:.2f}°).")
            filename.unlink()
            return

        # Cache only every Nth sample
        if self.msg_counter % self.pair_processing_frequency:
            rospy.loginfo(f"Ignoring image and pcl with seq {synced_data_filename_msg.header.seq}")
            filename.unlink()
        else:
            self.synced_data_filename_buffer.append([filename, synced_data_filename_msg.header])
            rospy.loginfo(
                f"Caching image and pcl with seq {synced_data_filename_msg.header.seq} "
                f"[cached {len(self.synced_data_filename_buffer)}/{self.number_image_pcl_pairs}]")
        self.msg_counter += 1

        # Reached the maximum capacity of the buffer.
        if len(self.synced_data_filename_buffer) == self.number_image_pcl_pairs:
            rospy.loginfo("Completed filling the buffer.")
            # No need to cache any more data
            self.synced_data_filename_subscriber.unregister()

    def _set_initial_transform_meta(self,
                                    initial_transform_meta_msg: UInt16MultiArrayStamped) -> None:
        # Sanity checks
        if len(initial_transform_meta_msg.data.data) == 1:
            assert len(self.initial_transform_meta) == 0
        elif len(initial_transform_meta_msg.data.data) == 2:
            assert len(self.initial_transform_meta) == 1
            assert initial_transform_meta_msg.data.data[0] == self.initial_transform_meta[0]
        else:
            assert False

        # Set the seq IDs used to obtain the initial transform
        self.initial_transform_meta = initial_transform_meta_msg.data.data

        # Remove all the cached data that will not be used
        final_delete_index = -1
        for index, [_, header] in enumerate(self.synced_data_filename_buffer):
            if header.seq < self.initial_transform_meta[0]:
                final_delete_index = index
            else:
                break
        if final_delete_index != -1:
            log = "Deleted cache up to including seq "
            log += f"{self.synced_data_filename_buffer[final_delete_index][1].seq} "
            # Delete files
            for filename, _ in self.synced_data_filename_buffer[:final_delete_index + 1]:
                filename.unlink()
            del self.synced_data_filename_buffer[:final_delete_index + 1]
            self.msg_counter = 0
            rospy.loginfo(
                log +
                f"[cached {len(self.synced_data_filename_buffer)}/{self.number_image_pcl_pairs}]")

        # In case something went wrong in the sequence of received messages
        if self.model is not None and self.initial_transform is not None:
            self._process_buffer()

    def _set_initial_transform(self, initial_transform_msg: TransformStamped) -> None:
        # Set the initial calibration received from the optimizer node
        translation = initial_transform_msg.transform.translation
        rotation = initial_transform_msg.transform.rotation
        translation = torch.Tensor([translation.x, translation.y, translation.z])
        rotation = torch.from_numpy(
            R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_matrix())

        print("Received T", translation.numpy())
        print("Received R (quat)  ", R.from_matrix(rotation.numpy()).as_quat())
        print("Received R (rotvec)", R.from_matrix(rotation.numpy()).as_rotvec(degrees=True))

        self.initial_transform = torch.eye(4)
        self.initial_transform[:3, 3] = translation
        self.initial_transform[:3, :3] = rotation

        # If the model has already been initialized, process the buffer
        if self.model is not None and len(self.initial_transform_meta) == 2:
            self._process_buffer()

    def _process_buffer(self) -> None:
        for [filename, _] in tqdm(self.synced_data_filename_buffer,
                                  total=len(self.synced_data_filename_buffer),
                                  desc="Generating 2D-3D correpondences"):

            image_msg, point_cloud_msg = None, None
            with rosbag.Bag(filename, "r") as bag:
                for topic, msg, _ in bag.read_messages(topics=["camera_img", "lidar_pcl"]):
                    if topic == "camera_img":
                        image_msg = msg
                    else:
                        point_cloud_msg = msg

            correspondences_img, correspondences_pcl, uncertainties = \
                self._predict_correspondences(image_msg, point_cloud_msg)

            if self.use_uncertainty and uncertainties is None:
                rospy.logwarn("'use_uncertainty' is set to True but uncertainty is not available "
                              "(warning is only printed once)")
                self.use_uncertainty = False
            if not self.use_uncertainty:
                uncertainties = None

            correspondences_img = correspondences_img.T
            correspondences_pcl = correspondences_pcl.T
            uncertainties = uncertainties.T if uncertainties is not None else None

            # Subsample correspondences
            number_of_samples = int(correspondences_img.shape[1] * self.amount_correspondences /
                                    100)
            if uncertainties is not None:
                probabilities = 1 / uncertainties
                probabilities /= probabilities.sum()
                keep_indices = self.rng.choice(correspondences_img.shape[1],
                                               number_of_samples,
                                               replace=False,
                                               shuffle=False,
                                               p=probabilities)
            else:
                keep_indices = self.rng.choice(correspondences_img.shape[1],
                                               number_of_samples,
                                               replace=False,
                                               shuffle=False)
            correspondences_img = correspondences_img[:, keep_indices]
            correspondences_pcl = correspondences_pcl[:, keep_indices]

            # Fill the message with the correspondences
            correspondences_msg = ImagePclCorrespondencesStamped()
            # Image pixels
            correspondences_msg.image.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
            correspondences_msg.image.layout.dim[0].label = "channels"
            correspondences_msg.image.layout.dim[0].size = correspondences_img.shape[0]
            correspondences_msg.image.layout.dim[0].stride = correspondences_img.size
            correspondences_msg.image.layout.dim[1].label = "samples"
            correspondences_msg.image.layout.dim[1].size = correspondences_img.shape[1]
            correspondences_msg.image.layout.dim[1].stride = correspondences_img.shape[1]
            correspondences_msg.image.data = correspondences_img.reshape(
                [correspondences_img.size]).astype(np.uintc).tolist()
            # Corresponding points in the point cloud
            correspondences_msg.pcl.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
            correspondences_msg.pcl.layout.dim[0].label = "channels"
            correspondences_msg.pcl.layout.dim[0].size = correspondences_pcl.shape[0]
            correspondences_msg.pcl.layout.dim[0].stride = correspondences_pcl.size
            correspondences_msg.pcl.layout.dim[1].label = "samples"
            correspondences_msg.pcl.layout.dim[1].size = correspondences_pcl.shape[1]
            correspondences_msg.pcl.layout.dim[1].stride = correspondences_pcl.shape[1]
            correspondences_msg.pcl.data = correspondences_pcl.reshape([correspondences_pcl.size
                                                                        ]).tolist()

            self.correspondences_publisher.publish(correspondences_msg)

        # Terminate node
        rospy.loginfo("Done processing. Shutting down CMRNext node.")
        rospy.signal_shutdown("Done processing.")

    def _predict_correspondences(
            self, image_msg: Image,
            point_cloud_msg: PointCloud2) -> Tuple[ArrayLike, ArrayLike, Optional[ArrayLike]]:
        assert self.model is not None
        assert self.initial_transform is not None

        # Convert messages
        image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
        image = torch.tensor(np.asarray(image)).float()
        point_cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(point_cloud_msg)
        # point_cloud = self.pointcloud2_to_numpy(point_cloud_msg)
        point_cloud = torch.tensor(point_cloud).float()
        point_cloud = torch.cat([point_cloud, torch.ones(point_cloud.shape[0], 1)], dim=1)

        # Employ CMRNext
        (correspondences_img,
         correspondences_pcl), uncertainties, final_calib = self.model.process_pair(
             image, point_cloud, self.initial_transform)
        self.final_calibs.append(final_calib)
        # Compute final average extrinsic calibration predicted by
        # CMRNext, used for comparison only
        if len(self.final_calibs) == len(self.synced_data_filename_buffer):
            self.final_calibs = torch.stack(self.final_calibs)
            quat_list = []
            for calib in self.final_calibs:
                quat_list.append(quaternion_from_matrix(calib).cpu().numpy())

            quat_list = np.stack(quat_list)

            average_t = self.final_calibs[:, :3, -1].median(0)[0]
            average_r = quat2mat(torch.tensor(average_quaternions(quat_list)))
            average_r[:3, -1] = average_t
            print(average_r)
        # Only keep x,y,z data in the point cloud
        correspondences_pcl = correspondences_pcl[:, :3]
        correspondences_img = correspondences_img.cpu().numpy()
        correspondences_pcl = correspondences_pcl.cpu().numpy()
        uncertainties = uncertainties.cpu().numpy() if uncertainties is not None else None
        return correspondences_img, correspondences_pcl, uncertainties

    def initialize_node(self,
                        name: str = "x",
                        camera_info_topic: str = "/camera_undistorted/camera_info",
                        synced_data_filename_topic: str = "/synced_data/filename",
                        initial_transform_topic: str = "/optimizer/initial_transform",
                        initial_transform_meta_topic: str = "/optimizer/initial_transform_meta",
                        correspondences_topic: str = "/cmrnext/correspondences") -> None:
        rospy.init_node(name)
        rospy.loginfo("Starting CMRNext node.")

        calibration_options_path = rospkg.RosPack().get_path("calib_cfg") + "/config/config.yaml"
        with open(calibration_options_path, "r", encoding="utf-8") as f:
            calibration_options = yaml.safe_load(f)
            self.number_image_pcl_pairs = \
                calibration_options["optimization"]["number_image_pcl_pairs"]
            self.amount_correspondences = \
                calibration_options["cmrnext"]["amount_correspondences"]
            self.rotation_threshold = \
                calibration_options["cmrnext"]["rotation_threshold"]

            # Compute the processing frequency based on the expected number of pair candidates
            self.pair_processing_frequency = \
                calibration_options["optimization"]["number_poses"] // self.number_image_pcl_pairs
            self.pair_processing_frequency -= 1  # Safety margin

        self.correspondences_publisher = rospy.Publisher(correspondences_topic,
                                                         ImagePclCorrespondencesStamped,
                                                         queue_size=10)

        self.synced_data_filename_subscriber = message_filters.Subscriber(
            synced_data_filename_topic, StringStamped)
        self.synced_data_filename_subscriber.registerCallback(self._cache_img_pcl_from_rosbag)

        self.initial_transform_meta_subscriber = message_filters.Subscriber(
            initial_transform_meta_topic, UInt16MultiArrayStamped)
        self.initial_transform_data_subscriber = message_filters.Subscriber(
            initial_transform_topic, TransformStamped)

        self.initial_transform_meta_subscriber.registerCallback(self._set_initial_transform_meta)
        self.initial_transform_data_subscriber.registerCallback(self._set_initial_transform)

        rospy.loginfo("Waiting for camera info message...")
        camera_info_msg = rospy.wait_for_message(camera_info_topic, CameraInfo)
        self._initialize_model(camera_info_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",
                        nargs="+",
                        type=str,
                        required=True,
                        help="List of the CMRNext weights.")
    parser.add_argument("--downsample", action="store_true", required=False, default=False)
    parser.add_argument("--image_shape", nargs="+", type=int, help="Height, width")
    # In case CMRNext is started with "roslaunch", ros might add __name and __log parameters at the
    # end, so we need to add this two optional args
    parser.add_argument("name", nargs="?", type=str, default="CMRNext", help="Name of the node")
    parser.add_argument("log", nargs="?", type=str, default="", help="Name of the node")
    args = parser.parse_args()
    # In case CMRNext is started with "roslaunch", AND the parameter --weights is specified,
    # the __name and __log will be appended to the args.weights list
    node_name = "cmrnext"
    # pylint: disable-next=consider-using-enumerate
    for i in range(len(args.weights)):
        if "__name" in args.weights[i] or "__log" in args.weights[i]:
            if "__name" in args.weights[i]:
                node_name = args.weights[i][8:]
            args.weights = args.weights[:i]
            break
    if "__name" in args.name:
        node_name = args.name[8:]

    target_shape_ = [args.image_shape[0], args.image_shape[1]]
    if target_shape_[0] % 64 > 0:
        target_shape_[0] = 64 * ((target_shape_[0] // 64) + 1)
    if target_shape_[1] % 64 > 0:
        target_shape_[1] = 64 * ((target_shape_[1] // 64) + 1)

    node = CMRNextNode(args.weights, tuple(target_shape_), args.downsample)
    node.initialize_node(node_name)
    rospy.spin()
