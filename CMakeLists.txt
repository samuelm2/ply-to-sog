cmake_minimum_required(VERSION 3.14)
project(ply-to-sog)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenMP)

# LibWebP
find_package(PkgConfig REQUIRED)
pkg_check_modules(WebP REQUIRED libwebp)

# Create the executable with the requested name
add_executable(ply-to-sog
    main.cpp
    SogEncoder.cpp
    dependencies/miniz.c
)

# Target-specific include directories
target_include_directories(ply-to-sog PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${WebP_INCLUDE_DIRS}
    "dependencies" # For json.hpp and nanoflann.hpp
)

# OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(ply-to-sog PRIVATE OpenMP::OpenMP_CXX)
endif()

# Link libraries
target_link_libraries(ply-to-sog PRIVATE
    Eigen3::Eigen
    ${WebP_LIBRARIES}
    stdc++fs # Required for std::filesystem on some older compilers
)
