# Updated Makefile for CUDA compilation

CXX = nvcc
CUDAFLAGS = -arch=sm_35
CUDADEBUGFLAGS = --generate-line-info
CPPFLAGS =  -std=c++14 --compiler-options -Wall
CPPDEBUGFLAGS = -g

load = source $$MODULESHOME/init/bash; module load cuda/11.0.3; module load gcc/8.3.0;

main = dsmc_gpu

# From cudaSamples
# Common includes and paths for CUDA
INCLUDES  := -I../../common/inc -I/usr/include/GL
LIBRARIES := 

################################################################################

# Makefile include to help find GL Libraries
include ./findgllib.mk

# OpenGL specific libraries
LIBRARIES += $(GLLINK)
LIBRARIES += -lGL -lGLU -lglut #-lm

all: $(main)
gpu_includes = 

dsmc_gpu: dsmc_gpu.cu $(gpu_includes)
	$(load) $(CXX) $(CUDAFLAGS) $(CUDADEBUGFLAGS) $(INCLUDES) -o bin/$@ $(CPPFLAGS) $(CPPDEBUGFLAGS) $^ 

dsmc_gpu_graphics: dsmc_gpu_graphics.cu $(gpu_includes)
	$(load) $(CXX) $(CUDAFLAGS) $(CUDADEBUGFLAGS) $(INCLUDES) -o bin/$@ $(CPPFLAGS) $(CPPDEBUGFLAGS) $^ $(LIBRARIES)


dsmc_orig: 	dsmc_orig.cc
	$(CXX) $(CUDAFLAGS) $(INCLUDES)  -o bin/$@  $(CPPFLAGS) $(CPPDEBUGFLAGS) $^ 

dsmc_orig_graphics: 	dsmc_orig_graphics.cc
	$(CXX) $(CUDAFLAGS) $(INCLUDES)  -o bin/$@  $(CPPFLAGS) $(CPPDEBUGFLAGS) $^ $(LIBRARIES)


clean:
	rm -f dsmc_gpu dsmc_orig dsmc_orig_graphics dsmc_gpu_graphics

