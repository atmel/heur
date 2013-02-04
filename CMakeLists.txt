CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(HeuristicsGPU)

# enable/disable compilation with nvcc, also propagates to code as #define

  option(USE_GPU "Toogle compilation with nvcc (Default: OFF)" OFF)
  
# cmake dirs
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#set(CXX_FLAGS "-frepo")

FIND_PACKAGE(CUDA)
if(CUDA_FOUND)
    message("Cuda found version: ${CUDA_VERSION_STRING}\n")

set( CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )

include_directories( /usr/local/cuda/include )

set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGs} -arch=sm_21 -keep --ptxas-options=-v)

  #LINK_LIBRARIES("-L/usr/local/cuda/lib -lcutil")

  #CUDA_ADD_EXECUTABLE(test main.cpp)

else(CUDA_FOUND)
   message("No CUDA found")
endif()     

  
macro(add_cpu_gpu_exe EXENAME FILENAME)
  if(USE_GPU)
	cuda_add_executable(${EXENAME} ${FILENAME}.cu)
  else(USE_GPU)
	add_executable(${EXENAME} ${FILENAME}.cpp)
  endif()
endmacro(add_cpu_gpu_exe)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/helper)
include_directories(${CMAKE_SOURCE_DIR}/src/abstractMethod)
include_directories(${CMAKE_SOURCE_DIR}/src/masterMethod)
include_directories(${CMAKE_SOURCE_DIR}/src/slaveMethod)
include_directories(${CMAKE_SOURCE_DIR}/src/container)
include_directories(${CMAKE_SOURCE_DIR}/src/provider)
  
# subdirs
add_subdirectory(progs)






                                          