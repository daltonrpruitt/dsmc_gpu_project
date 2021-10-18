# Updated Makefile for CUDA compilation

CXX = nvcc
include_flags = -I.
CUDAFLAGS = -arch=sm_35
CUDADEBUGFLAGS = --generate-line-info
CPPFLAGS =  -std=c++14 --compiler-options -Wall
CPPDEBUGFLAGS = -g

load = source $$MODULESHOME/init/bash; module load cuda/11.0.3; module load gcc/8.3.0;

main = dsmc_gpu


all: $(main)
gpu_includes = 

$(main): $(main).cu $(gpu_includes)
	$(load) $(CXX) $(CUDAFLAGS) $(CUDADEBUGFLAGS) $(include_flags) -o bin/$@ $(CPPFLAGS) $(CPPDEBUGFLAGS) $^
 

dsmc_orig: 	dsmc_orig.cc
	$(CXX) $(CUDAFLAGS) $(include_flags)  -o bin/$@  $(CPPFLAGS) $(CPPDEBUGFLAGS) $^


clean:


