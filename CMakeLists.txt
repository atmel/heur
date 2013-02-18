CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(HeuristicsGPU)

# enable/disable compilation with nvcc, also propagates to code as #define

  option(USE_GPU "Toogle compilation with nvcc (Default: OFF)" OFF)
  
# cmake dirs
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# -pg is for profiling
set(CMAKE_CXX_FLAGS "-O3 -ffast-math -pg")
#set(CC_FLAGS )

FIND_PACKAGE(CUDA)
if(CUDA_FOUND)
    message("Cuda found version: ${CUDA_VERSION_STRING}\n")

set( CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin )
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin )

include_directories( /usr/local/cuda/include )

set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGs} -arch=sm_21 -keep --ptxas-options=-v)

LINK_LIBRARIES("-lrt")  #time measurement

else(CUDA_FOUND)
   message("No CUDA found")
endif()     

  
  
# changes file extension to EX2 if PATHTOFILE.EX1 exists and 
# PATHTOFILE.EX2 does not to prevent overwriting
macro(change_file_extension PATHTOFILE EX1 EX2)
	if(EXISTS ${PATHTOFILE}.${EX1} AND NOT EXISTS ${PATHTOFILE}.${EX2})
		execute_process(COMMAND mv ${PATHTOFILE}.${EX1} ${PATHTOFILE}.${EX2})
		message("changing ${PATHTOFILE}.${EX1} --> ${PATHTOFILE}.${EX2}")
	endif()
endmacro(change_file_extension)
  
#changes file extension to necessary form
macro(add_cpu_gpu_exe EXENAME FILENAME)
  if(USE_GPU)
	change_file_extension(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME} cpp cu)
	cuda_add_executable(${EXENAME} ${FILENAME}.cu)
	
  else(USE_GPU)
	change_file_extension(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME} cu cpp)
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






                                          