cmake_minimum_required (VERSION 3.8)
project ("SecurityCam")

# --- UPDATE THESE PATHS ---
# 1. Where did you put OpenCV? (Use forward slashes / )
set(OpenCV_DIR "D:/opencv/build") 

# 2. Where is your NCNN? (Copy path from your specific download folder)
set(ncnn_DIR "C:/Users/songo/Downloads/ncnn-20250916-windows-vs2022/x64/lib/cmake/ncnn")
# --------------------------

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_executable (SecurityCam "main.cpp")
# Tell CMake where to find net.h
target_include_directories(SecurityCam PRIVATE ${ncnn_INCLUDE_DIRS})
# Add ncnn to the list of libraries
target_link_libraries(SecurityCam PRIVATE ${OpenCV_LIBS} ncnn)