# The name of the project
PROJECT(light_sources)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6.4)
set(CMAKE_MODULE_PATH ${light_sources_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(EXECUTABLE_OUTPUT_PATH ${light_sources_BINARY_DIR})

## Searching CUDA
#FIND_PACKAGE(CUDA REQUIRED)

# Searching OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(SYSTEM ${OpenCV_INCLUDE_DIRS})

#FIND_PACKAGE(Boost)
#IF (Boost_FOUND)
#  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
#  ADD_DEFINITIONS( "-DHAS_BOOST" )
#ENDIF()

FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(GLUT REQUIRED)
include(FindPkgConfig)
pkg_check_modules(GLEW glew)
if(NOT GLEW_FOUND)
  find_package(GLEW REQUIRED)
  if(GLEW_FOUND)
    message(STATUS "But fallback FindGLEW.cmake works")
  endif()
endif()
include_directories(${GLEW_INCLUDE_DIRS})

FIND_PACKAGE(LPSolve)
IF (LPSOLVE_FOUND)
  INCLUDE_DIRECTORIES(${LPSOLVE_INCLUDE_DIR})
ENDIF()

# Compiler setup
set(CMAKE_BUILD_TYPE debug)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x ${VALID_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")

# Adding src directory to the project
ADD_SUBDIRECTORY(src)
