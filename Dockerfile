FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 AS base

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO noetic

RUN \
    # Update nvidia GPG key
    rm /etc/apt/sources.list.d/cuda.list && \
    apt-key del 7fa2af80 && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

# preseed tzdata, update package index, upgrade packages and install needed software
RUN truncate -s0 /tmp/preseed.cfg; \
    echo "tzdata tzdata/Areas select Europe" >> /tmp/preseed.cfg; \
    echo "tzdata tzdata/Zones/Europe select Berlin" >> /tmp/preseed.cfg; \
    debconf-set-selections /tmp/preseed.cfg && \
    rm -f /etc/timezone /etc/localtime && \
    apt-get update && \
    apt-get install -y tzdata

# ROS
# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends dirmngr gnupg2
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list
RUN apt-get update && apt-get install -y --no-install-recommends ros-noetic-ros-core=1.5.0-1*
RUN apt-get update && apt-get install -y --no-install-recommends nano build-essential git byobu curl xclip

# ceres dependencies
RUN apt-get update && apt-get install -y libgoogle-glog-dev libgflags-dev libatlas-base-dev libsuitesparse-dev libeigen3-dev

#PCL
RUN apt-get update && apt-get install -y --no-install-recommends libpcl-dev ros-noetic-pcl-ros ros-noetic-pcl-conversions

#OPENCV
RUN apt-get update && apt-get install -y --no-install-recommends libopencv-dev python3-opencv ros-noetic-cv-bridge

# Additional ROS packages
RUN apt-get update && apt-get install -y ros-noetic-rviz python3-catkin-tools ros-noetic-tf-conversions ros-noetic-tf2-sensor-msgs ros-noetic-image-transport-plugins ros-noetic-rqt-image-view ros-noetic-message-filters python3-rospy python3-message-filters python3-sensor-msgs python3-pip ros-noetic-geodesy ros-noetic-nmea-msgs ros-noetic-libg2o
RUN apt-get update && apt-get install -y ros-noetic-ros-numpy
RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

RUN apt-get update && apt-get install -y unzip ros-noetic-hector-trajectory-server ros-noetic-image-pipeline

RUN apt-get clean && rm -rf /var/lib/apt/lists/*
# cleanup of files from setup
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# We will now build the individual dependencies, all of those build up independently from the base image. Thus, we processes
# will run in parallel. If one of the components did not change it will not be rebuilt. Note that whenever something is added to the
# base image (everything above), everything dependent on that has to be rebuild.
FROM base AS pangolin_build
WORKDIR /root
RUN git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
WORKDIR /root/Pangolin
RUN git reset --hard v0.6
RUN mkdir -p build && cd build && cmake ..
RUN cd build && cmake --build .

FROM base AS opencv_build
WORKDIR /root/opencv
RUN wget https://github.com/opencv/opencv/archive/4.7.0.zip
RUN unzip 4.7.0.zip
RUN git clone https://github.com/opencv/opencv_contrib.git
WORKDIR /root/opencv/opencv_contrib
RUN git checkout 4.7.0
# Remove all unnecessary modules
RUN rm -rf ./modules/alphamat/ ./modules/aruco/ ./modules/barcode/ ./modules/bgsegm/ ./modules/bioinspired/ ./modules/ccalib/ ./modules/cnn_3dobj/ ./modules/cudabgsegm/ ./modules/cudacodec/ ./modules/cudafeatures2d/ ./modules/cudaobjdetect/ ./modules/cudastereo/ ./modules/cvv/ ./modules/datasets/ ./modules/dnn_objdetect/ ./modules/dnn_superres/ ./modules/dnns_easily_fooled/ ./modules/dpm/ ./modules/face/ ./modules/freetype/ ./modules/fuzzy/ ./modules/hdf/ ./modules/hfs/ ./modules/img_hash/ ./modules/intensity_transform/ ./modules/julia/ ./modules/line_descriptor/ ./modules/matlab/ ./modules/mcc/ ./modules/ovis/ ./modules/phase_unwrapping/ ./modules/quality/ ./modules/rapid/ ./modules/README.md ./modules/reg/ ./modules/rgbd/ ./modules/saliency/ ./modules/sfm/ ./modules/shape/ ./modules/stereo/ ./modules/structured_light/ ./modules/superres/ ./modules/surface_matching/ ./modules/text/ ./modules/videostab/ ./modules/viz/ ./modules/wechat_qrcode/ ./modules/xfeatures2d/ ./modules/xobjdetect/ ./modules/xphoto/
COPY ./pythoncuda /root/opencv/opencv_contrib/modules/pythoncuda
WORKDIR /root/opencv
RUN mkdir -p build && cd build && cmake -DOPENCV_EXTRA_MODULES_PATH=/root/opencv/opencv_contrib/modules -DCMAKE_BUILD_TYPE=RELEASE -D WITH_TBB=ON  -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages -D CUDA_ARCH_BIN=5.0,5.2,6.1,7.0,7.5,8.0,8.6 -DCUDA_ARCH_PTX=5.2 ../opencv-4.7.0/
RUN cd build && cmake --build . -j $(nproc)

#CERES
FROM base AS ceres_solver_build
WORKDIR /root/
RUN git clone https://ceres-solver.googlesource.com/ceres-solver
RUN mkdir /root/ceres-solver/bin
WORKDIR /root/ceres-solver
RUN git checkout 2.1.0
WORKDIR /root/ceres-solver/bin
RUN cmake ..
RUN make -j4

FROM base AS orb_slam3_dependencies
WORKDIR /root/
RUN git clone https://github.com/UZ-SLAMLab/ORB_SLAM3
WORKDIR /root/ORB_SLAM3/
RUN mkdir -p Thirdparty/DBoW2/build && cd Thirdparty/DBoW2/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
RUN mkdir -p Thirdparty/g2o/build && cd Thirdparty/g2o/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
RUN mkdir -p Thirdparty/Sophus/build && cd Thirdparty/Sophus/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8
RUN cd Vocabulary && tar -xf ORBvoc.txt.tar.gz

FROM base AS building_calibration
# we now copy the build files from the individual dependencies to the final docker container. This will wait for all components to be built
COPY --from=ceres_solver_build /root/ceres-solver /root/ceres-solver
COPY --from=pangolin_build /root/Pangolin/ /root/Pangolin/
COPY --from=opencv_build /root/opencv/ /root/opencv/
COPY --from=orb_slam3_dependencies /root/ORB_SLAM3/ /root/ORB_SLAM3/
RUN cd /root/Pangolin/build && cmake --install .
RUN cd /root/opencv/build && cmake --install .
RUN cd /root/ceres-solver/bin && make install
RUN cd /root/ORB_SLAM3/ && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j8

WORKDIR /
RUN touch /root/.bashrc
RUN echo ". /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo ". /root/catkin_ws/devel/setup.bash" >> /root/.bashrc


RUN mkdir -p /root/catkin_ws/src
# HDL Graph Slam requirements
WORKDIR /root/catkin_ws/src
RUN git clone https://github.com/koide3/ndt_omp.git
RUN git clone https://github.com/SMRT-AIST/fast_gicp.git --recursive

# Image Undistort
RUN git clone https://github.com/ethz-asl/image_undistort.git

# Additional dependencies for CMRNet
RUN pip3 install numpy==1.20.3 scikit-image pyquaternion mathutils==2.81.2 tqdm python-dateutil==2.8.2 open3d pillow==10.3.0

# Create catkin workspace and copy calibration codes
WORKDIR /root/catkin_ws
RUN mkdir -p /root/catkin_ws/src
COPY ./src /root/catkin_ws/src/mdpcalib

# Python3 setup for CMRNet
# WORKDIR /root/catkin_ws/src/mdpcalib/CMRNet
# RUN pip3 install -r requirements.txt
# WORKDIR /root/catkin_ws/src/mdpcalib/CMRNet/visibility_pkg
# RUN python3 setup.py install

# Build catkin workspace
WORKDIR /root/catkin_ws
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin config --profile default --cmake-args -DCMAKE_BUILD_TYPE=Release'
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin config --profile debug -x _debug --cmake-args -DCMAKE_BUILD_TYPE=Debug'
RUN /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin build -cs'

# Setup environment and entrypoint
RUN touch /ros_entrypoint.sh
# RUN sed -i "6i source \"/root/catkin_ws/devel/setup.bash\"" /ros_entrypoint.sh
RUN echo "#!/bin/bash" >> /ros_entrypoint.sh
RUN echo "set -e" >> /ros_entrypoint.sh
RUN echo "source \"/opt/ros/noetic/setup.bash\" --" >> /ros_entrypoint.sh
RUN echo "source \"/root/catkin_ws/devel/setup.bash\"" >> /ros_entrypoint.sh
RUN echo "exec \"\$@\"" >> /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

WORKDIR /
SHELL ["bash", "--command"]
ENV SHELL /usr/bin/bash
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
