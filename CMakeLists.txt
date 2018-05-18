cmake_minimum_required(VERSION 3.8)
project(nnt)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH})

include(CheckCXXCompilerFlag)
include(cmake/utils.cmake)

#
# compile flags
#
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

#
# find packages
#
find_package(Threads)
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)

#
# include directories
#
include_directories(SYSTEM ${FLATBUFFERS_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

#
# list cpp files
#
file(GLOB SRCS "${CMAKE_SOURCE_DIR}/src/*.cc")

#
# create tensorflow proto library
#
add_executable(nnt ${SRCS})

target_link_libraries(nnt
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  ${Boost_LIBRARIES}
  /usr/local/lib/libflatbuffers.a)
