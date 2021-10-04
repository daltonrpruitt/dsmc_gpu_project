# Updated Makefile for CUDA compilation

CXX = nvcc
include_flags = -I.
CUDAFLAGS = -arch=sm_35
CUDADEBUGFLAGS = --generate-line-info
CPPFLAGS =  -std=c++11 --compiler-options -Wall
CPPDEBUGFLAGS = -g

main = dsmc_gpu

load = source $$MODULESHOME/init/bash; module load cuda/11.0.3; module load gcc/8.3.0;

all: $(main)

$(main): $(main).cc 
	$(CXX) $(CUDAFLAGS) $(CUDADEBUGFLAGS) $(include_flags) -o bin/$@ $(CPPFLAGS) $^
 

dsmc_orig: 	dsmc_orig.cc
	$(CXX) $(CUDAFLAGS) $(include_flags)  -o bin/$@  $(CPPFLAGS)  $^


clean:


