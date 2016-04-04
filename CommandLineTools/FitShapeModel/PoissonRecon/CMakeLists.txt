cmake_minimum_required(VERSION 2.8)

PROJECT(PoissonRecon)

include_directories( "../vcglib" )

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11" )

add_library(PoissonRecon 
  CmdLineParser.cpp
  Factor.cpp
  Geometry.cpp
  MarchingCubes.cpp
  PlyFile.cpp
  NormalEstimator.cpp
  PoissonRecon.cpp)
