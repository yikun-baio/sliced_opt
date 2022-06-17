cmake_minimum_required(VERSION 3.21)
project(C___Code)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -Wextra -pedantic -pedantic-errors -g")

find_package(xtensor REQUIRED)

set (SLICED_OPT_EXPERIMENT
        experiments/main.cpp
        include/helpers.h
        include/sliced_opt.h
        src/helpers.cpp
        src/sliced_opt.cpp
        )
add_executable(slicedOpt ${SLICED_OPT_EXPERIMENT})


target_include_directories(slicedOpt PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(slicedOpt PUBLIC xtensor)