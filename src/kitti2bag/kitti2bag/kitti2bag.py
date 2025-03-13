#!env python
# -*- coding: utf-8 -*-

import sys

try:
    import pykitti
except ImportError as e:
    print('Could not load module \'pykitti\'. Please run `pip install pykitti`')
    sys.exit(1)

import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import progressbar
import rosbag
import rospy
import sensor_msgs.point_cloud2 as pcl2
import tf
from cv_bridge import CvBridge
from geometry_msgs.msg import Transform, TransformStamped, TwistStamped
from sensor_msgs.msg import CameraInfo, Imu, NavSatFix, PointField
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage


def save_imu_data(bag, kitti, imu_frame_id, topic):
    print('Exporting IMU')
    sequence = 0
    for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
        q = tf.transformations.quaternion_from_euler(oxts.packet.roll, oxts.packet.pitch,
                                                     oxts.packet.yaw)
        imu = Imu()
        imu.header.frame_id = imu_frame_id
        imu.header.stamp = rospy.Time.from_sec(float(timestamp.strftime('%s.%f')))
        imu.header.seq = sequence
        imu.orientation.x = q[0]
        imu.orientation.y = q[1]
        imu.orientation.z = q[2]
        imu.orientation.w = q[3]
        imu.linear_acceleration.x = oxts.packet.af
        imu.linear_acceleration.y = oxts.packet.al
        imu.linear_acceleration.z = oxts.packet.au
        imu.angular_velocity.x = oxts.packet.wf
        imu.angular_velocity.y = oxts.packet.wl
        imu.angular_velocity.z = oxts.packet.wu
        bag.write(topic, imu, t=imu.header.stamp)
        sequence += 1


def save_dynamic_tf(bag, kitti, kitti_type, initial_time):
    print('Exporting time dependent transformations')
    if kitti_type.find('raw') != -1:
        for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
            tf_oxts_msg = TFMessage()
            tf_oxts_transform = TransformStamped()
            tf_oxts_transform.header.stamp = rospy.Time.from_sec(float(timestamp.strftime('%s.%f')))
            tf_oxts_transform.header.frame_id = 'world'
            tf_oxts_transform.child_frame_id = 'base_link'

            transform = (oxts.T_w_imu)
            t = transform[0:3, 3]
            q = tf.transformations.quaternion_from_matrix(transform)
            oxts_tf = Transform()

            oxts_tf.translation.x = t[0]
            oxts_tf.translation.y = t[1]
            oxts_tf.translation.z = t[2]

            oxts_tf.rotation.x = q[0]
            oxts_tf.rotation.y = q[1]
            oxts_tf.rotation.z = q[2]
            oxts_tf.rotation.w = q[3]

            tf_oxts_transform.transform = oxts_tf
            tf_oxts_msg.transforms.append(tf_oxts_transform)

            bag.write('/tf', tf_oxts_msg, tf_oxts_msg.transforms[0].header.stamp)

    elif kitti_type.find('odom') != -1:
        timestamps = map(lambda x: initial_time + x.total_seconds(), kitti.timestamps)
        for timestamp, tf_matrix in zip(timestamps, kitti.T_w_cam0):
            tf_msg = TFMessage()
            tf_stamped = TransformStamped()
            tf_stamped.header.stamp = rospy.Time.from_sec(timestamp)
            tf_stamped.header.frame_id = 'world'
            tf_stamped.child_frame_id = 'camera_left'

            t = tf_matrix[0:3, 3]
            q = tf.transformations.quaternion_from_matrix(tf_matrix)
            transform = Transform()

            transform.translation.x = t[0]
            transform.translation.y = t[1]
            transform.translation.z = t[2]

            transform.rotation.x = q[0]
            transform.rotation.y = q[1]
            transform.rotation.z = q[2]
            transform.rotation.w = q[3]

            tf_stamped.transform = transform
            tf_msg.transforms.append(tf_stamped)

            bag.write('/tf', tf_msg, tf_msg.transforms[0].header.stamp)


def save_camera_data(bag, kitti_type, kitti, util, bridge, camera, camera_frame_id, topic,
                     initial_time):
    print(f'Exporting camera {camera}')
    if kitti_type.find('raw') != -1:
        camera_pad = f'{camera:02}'
        image_dir = os.path.join(kitti.data_path, f'image_{camera_pad}')
        image_path = os.path.join(image_dir, 'data')
        image_filenames = sorted(os.listdir(image_path))
        with open(os.path.join(image_dir, 'timestamps.txt'), encoding='utf-8') as f:
            image_datetimes = map(lambda x: datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S.%f'),
                                  f.readlines())

        calib = CameraInfo()
        calib.header.frame_id = camera_frame_id
        calib.width, calib.height = tuple(util[f'S_rect_{camera_pad}'].tolist())
        calib.distortion_model = 'plumb_bob'
        calib.K = util[f'K_{camera_pad}']
        calib.R = util[f'R_rect_{camera_pad}']
        calib.D = util[f'D_{camera_pad}']
        calib.P = util[f'P_rect_{camera_pad}']

    elif kitti_type.find('odom') != -1:
        camera_pad = f'{camera:01}'
        image_path = os.path.join(kitti.sequence_path, f'image_{camera_pad}')
        image_filenames = sorted(os.listdir(image_path))
        image_datetimes = map(lambda x: initial_time + x.total_seconds(), kitti.timestamps)

        calib = CameraInfo()
        calib.header.frame_id = camera_frame_id
        calib.P = util[f'P{camera_pad}']

    iterable = zip(image_datetimes, image_filenames)
    pbar = progressbar.ProgressBar()
    sequence = 0
    for dt, filename in pbar(iterable):
        image_filename = os.path.join(image_path, filename)
        cv_image = cv2.imread(image_filename)
        calib.height, calib.width = cv_image.shape[:2]
        if camera in (0, 1):
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        encoding = 'mono8' if camera in (0, 1) else 'bgr8'
        image_message = bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
        image_message.header.frame_id = camera_frame_id
        image_message.header.seq = sequence
        if kitti_type.find('raw') != -1:
            image_message.header.stamp = rospy.Time.from_sec(float(datetime.strftime(dt, '%s.%f')))
            topic_ext = '/image_raw'
        elif kitti_type.find('odom') != -1:
            image_message.header.stamp = rospy.Time.from_sec(dt)
            topic_ext = '/image_rect'
        calib.header.stamp = image_message.header.stamp
        calib.header.seq = sequence
        bag.write(topic + topic_ext, image_message, t=image_message.header.stamp)
        bag.write(topic + '/camera_info', calib, t=calib.header.stamp)
        sequence += 1


def save_velo_data(bag, kitti, velo_frame_id, topic):
    print('Exporting velodyne data')
    velo_path = os.path.join(kitti.data_path, 'velodyne_points')
    velo_data_dir = os.path.join(velo_path, 'data')
    velo_filenames = sorted(os.listdir(velo_data_dir))
    with open(os.path.join(velo_path, 'timestamps.txt'), encoding='utf-8') as f:
        lines = f.readlines()
        velo_datetimes = []
        for line in lines:
            if len(line) == 1:
                continue
            dt = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            velo_datetimes.append(dt)

    iterable = zip(velo_datetimes, velo_filenames)
    pbar = progressbar.ProgressBar()
    sequence = 0
    for dt, filename in pbar(iterable):
        if dt is None:
            continue

        velo_filename = os.path.join(velo_data_dir, filename)

        # read binary data
        scan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)

        # get ring channel
        depth = np.linalg.norm(scan, 2, axis=1)
        pitch = np.arcsin(scan[:, 2] / depth)  # arcsin(z, depth)
        fov_down = -24.8 / 180.0 * np.pi
        fov = (abs(-24.8) + abs(2.0)) / 180.0 * np.pi
        proj_y = (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
        proj_y *= 64  # in [0.0, H]
        proj_y = np.floor(proj_y)
        proj_y = np.minimum(64 - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        proj_y = proj_y.reshape(-1, 1)
        scan = np.concatenate((scan, proj_y), axis=1)

        # add empty time channel
        time = np.zeros_like(proj_y)
        scan = np.concatenate((scan, time), axis=1)

        #
        scan = scan.tolist()
        for i, _ in enumerate(scan):
            scan[i][-2] = int(scan[i][-2])  # turn ring number into integer

        # create header
        header = Header()
        header.frame_id = velo_frame_id
        header.stamp = rospy.Time.from_sec(float(datetime.strftime(dt, '%s.%f')))
        header.seq = sequence

        # fill pcl msg
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
            PointField('ring', 16, PointField.UINT16, 1),
            PointField('time', 20, PointField.FLOAT32, 1)
        ]
        pcl_msg = pcl2.create_cloud(header, fields, scan)

        bag.write(topic + '/pointcloud', pcl_msg, t=pcl_msg.header.stamp)
        sequence += 1


def get_static_transform(from_frame_id, to_frame_id, transform):
    t = transform[0:3, 3]
    q = tf.transformations.quaternion_from_matrix(transform)
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = from_frame_id
    tf_msg.child_frame_id = to_frame_id
    tf_msg.transform.translation.x = float(t[0])
    tf_msg.transform.translation.y = float(t[1])
    tf_msg.transform.translation.z = float(t[2])
    tf_msg.transform.rotation.x = float(q[0])
    tf_msg.transform.rotation.y = float(q[1])
    tf_msg.transform.rotation.z = float(q[2])
    tf_msg.transform.rotation.w = float(q[3])
    return tf_msg


def inv(transform):
    'Invert rigid body transformation matrix'
    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    t_inv = -1 * R.T.dot(t)
    transform_inv = np.eye(4)
    transform_inv[0:3, 0:3] = R.T
    transform_inv[0:3, 3] = t_inv
    return transform_inv


def save_static_transforms(bag, transforms, timestamps):
    print('Exporting static transformations')
    tfm = TFMessage()
    for transform in transforms:
        t = get_static_transform(from_frame_id=transform[0],
                                 to_frame_id=transform[1],
                                 transform=transform[2])
        tfm.transforms.append(t)
    for timestamp in timestamps:
        time = rospy.Time.from_sec(float(timestamp.strftime('%s.%f')))
        for i, _ in enumerate(tfm.transforms):
            tfm.transforms[i].header.stamp = time
        bag.write('/tf_static', tfm, t=time)


def save_gps_fix_data(bag, kitti, gps_frame_id, topic):
    sequence = 0
    for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
        navsatfix_msg = NavSatFix()
        navsatfix_msg.header.frame_id = gps_frame_id
        navsatfix_msg.header.stamp = rospy.Time.from_sec(float(timestamp.strftime('%s.%f')))
        navsatfix_msg.header.seq = sequence
        navsatfix_msg.latitude = oxts.packet.lat
        navsatfix_msg.longitude = oxts.packet.lon
        navsatfix_msg.altitude = oxts.packet.alt
        navsatfix_msg.status.service = 1
        bag.write(topic, navsatfix_msg, t=navsatfix_msg.header.stamp)
        sequence += 1


def save_gps_vel_data(bag, kitti, gps_frame_id, topic):
    sequence = 0
    for timestamp, oxts in zip(kitti.timestamps, kitti.oxts):
        twist_msg = TwistStamped()
        twist_msg.header.frame_id = gps_frame_id
        twist_msg.header.stamp = rospy.Time.from_sec(float(timestamp.strftime('%s.%f')))
        twist_msg.header.seq = sequence
        twist_msg.twist.linear.x = oxts.packet.vf
        twist_msg.twist.linear.y = oxts.packet.vl
        twist_msg.twist.linear.z = oxts.packet.vu
        twist_msg.twist.angular.x = oxts.packet.wf
        twist_msg.twist.angular.y = oxts.packet.wl
        twist_msg.twist.angular.z = oxts.packet.wu
        bag.write(topic, twist_msg, t=twist_msg.header.stamp)
        sequence += 1


def run_kitti2bag():
    parser = argparse.ArgumentParser(
        description='Convert KITTI dataset to ROS bag file the easy way!')
    # Accepted argument values
    kitti_types = ['raw_synced', 'odom_color', 'odom_gray']
    odometry_sequences = []
    for s in range(22):
        odometry_sequences.append(str(s).zfill(2))

    parser.add_argument('kitti_type', choices=kitti_types, help='KITTI dataset type')
    parser.add_argument(
        'dir',
        nargs='?',
        default=os.getcwd(),
        help='base directory of the dataset, if no directory passed the default is the current one')
    parser.add_argument(
        '-t',
        '--date',
        help='date of the raw dataset (i.e. 2011_09_26), option is only for RAW datasets.')
    parser.add_argument(
        '-r',
        '--drive',
        help='drive number of the raw dataset (i.e. 0001), option is only for RAW datasets.')
    parser.add_argument(
        '-s',
        '--sequence',
        choices=odometry_sequences,
        help=
        'sequence of the odometry dataset (between 00 - 21), option is only for ODOMETRY datasets.')
    args = parser.parse_args()

    bridge = CvBridge()
    compression = rosbag.Compression.NONE
    # compression = rosbag.Compression.BZ2
    # compression = rosbag.Compression.LZ4

    # CAMERAS
    cameras = [(0, 'camera_gray_left', '/kitti/camera_gray_left'),
               (1, 'camera_gray_right', '/kitti/camera_gray_right'),
               (2, 'camera_color_left', '/kitti/camera_color_left'),
               (3, 'camera_color_right', '/kitti/camera_color_right')]

    if args.kitti_type.find('raw') != -1:

        if args.date is None:
            print('Date option is not given. It is mandatory for raw dataset.')
            print('Usage for raw dataset: kitti2bag raw_synced [dir] -t <date> -r <drive>')
            sys.exit(1)
        elif args.drive is None:
            print('Drive option is not given. It is mandatory for raw dataset.')
            print('Usage for raw dataset: kitti2bag raw_synced [dir] -t <date> -r <drive>')
            sys.exit(1)

        bag = rosbag.Bag(f'kitti_{args.date}_drive_{args.drive}_{args.kitti_type[4:]}.bag',
                         'w',
                         compression=compression)
        kitti = pykitti.raw(args.dir, args.date, args.drive)
        if not os.path.exists(kitti.data_path):
            print(f'Path {kitti.data_path} does not exists. Exiting.')
            sys.exit(1)

        if len(kitti.timestamps) == 0:
            print('Dataset is empty? Exiting.')
            sys.exit(1)

        try:
            # IMU
            imu_frame_id = 'imu_link'
            imu_topic = '/kitti/oxts/imu'
            gps_fix_topic = '/kitti/oxts/gps/fix'
            gps_vel_topic = '/kitti/oxts/gps/vel'
            velo_frame_id = 'velo_link'
            velo_topic = '/kitti/velo'

            T_base_link_to_imu = np.eye(4, 4)
            T_base_link_to_imu[0:3, 3] = [-2.71 / 2.0 - 0.05, 0.32, 0.93]

            # tf_static
            transforms = [('base_link', imu_frame_id, T_base_link_to_imu),
                          (imu_frame_id, velo_frame_id, inv(kitti.calib.T_velo_imu)),
                          (imu_frame_id, cameras[0][1], inv(kitti.calib.T_cam0_imu)),
                          (imu_frame_id, cameras[1][1], inv(kitti.calib.T_cam1_imu)),
                          (imu_frame_id, cameras[2][1], inv(kitti.calib.T_cam2_imu)),
                          (imu_frame_id, cameras[3][1], inv(kitti.calib.T_cam3_imu))]

            util = pykitti.utils.read_calib_file(
                os.path.join(kitti.calib_path, 'calib_cam_to_cam.txt'))

            # Export
            save_static_transforms(bag, transforms, kitti.timestamps)
            save_dynamic_tf(bag, kitti, args.kitti_type, initial_time=None)
            save_imu_data(bag, kitti, imu_frame_id, imu_topic)
            save_gps_fix_data(bag, kitti, imu_frame_id, gps_fix_topic)
            save_gps_vel_data(bag, kitti, imu_frame_id, gps_vel_topic)
            for camera in cameras:
                if 'gray' in camera[1]:
                    continue
                save_camera_data(bag,
                                 args.kitti_type,
                                 kitti,
                                 util,
                                 bridge,
                                 camera=camera[0],
                                 camera_frame_id=camera[1],
                                 topic=camera[2],
                                 initial_time=None)
            save_velo_data(bag, kitti, velo_frame_id, velo_topic)

        finally:
            print('## OVERVIEW ##')
            print(bag)
            bag.close()

    elif args.kitti_type.find('odom') != -1:

        if args.sequence is None:
            print('Sequence option is not given. It is mandatory for odometry dataset.')
            print(
                'Usage for odometry dataset: kitti2bag {odom_color, odom_gray} [dir] -s <sequence>')
            sys.exit(1)

        bag = rosbag.Bag(f'kitti_data_odometry_{args.kitti_type[5:]}_sequence_{args.sequence}.bag',
                         'w',
                         compression=compression)

        kitti = pykitti.odometry(args.dir, args.sequence)
        if not os.path.exists(kitti.sequence_path):
            print('Path {kitti.sequence_path} does not exists. Exiting.')
            sys.exit(1)

        kitti.load_calib()
        kitti.load_timestamps()

        if len(kitti.timestamps) == 0:
            print('Dataset is empty? Exiting.')
            sys.exit(1)

        if args.sequence in odometry_sequences[:11]:
            print('Odometry dataset sequence {args.sequence} has ground truth information (poses).')
            kitti.load_poses()

        try:
            util = pykitti.utils.read_calib_file(
                os.path.join(args.dir, 'sequences', args.sequence, 'calib.txt'))
            current_epoch = (datetime.utcnow() - datetime(1970, 1, 1)).total_seconds()
            # Export
            if args.kitti_type.find('gray') != -1:
                used_cameras = cameras[:2]
            elif args.kitti_type.find('color') != -1:
                used_cameras = cameras[-2:]

            save_dynamic_tf(bag, kitti, args.kitti_type, initial_time=current_epoch)
            for camera in used_cameras:
                save_camera_data(bag,
                                 args.kitti_type,
                                 kitti,
                                 util,
                                 bridge,
                                 camera=camera[0],
                                 camera_frame_id=camera[1],
                                 topic=camera[2],
                                 initial_time=current_epoch)

        finally:
            print('## OVERVIEW ##')
            print(bag)
            bag.close()
