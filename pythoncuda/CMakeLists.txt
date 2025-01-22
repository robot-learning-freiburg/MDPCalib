set(the_description "access opencv cuda methods from Python")

# find_package(cuda REQUIRED)
include_directories("/usr/local/cuda/targets/x86_64-linux/include/")

ocv_define_module(pythoncuda opencv_video opencv_core opencv_imgproc opencv_tracking opencv_cudaoptflow opencv_cudaarithm opencv_cudaimgproc opencv_cudafilters opencv_cudalegacy opencv_objdetect opencv_calib3d opencv_cudev WRAP python)

ocv_module_include_directories(opencv_core)
