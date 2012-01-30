#
# Build script for project
#

# Add source files here
EXECUTABLE	     := fmr
# Cuda source files (compiled with cudacc)
CUFILES		     := particleSystem.cu new_kcall.cu 
CUDEPS		     := particles_kernel.cu particles_kernel.cuh particleSystem.cuh \
					connectedgraphs.h new_kcall.cuh new_kern.cu new_kern.cuh 
CCFILES		     := particles.cpp particleSystem.cpp render_particles.cpp shaders.cpp connectedgraphs.cpp

CUDACCFLAGS := #-DTHRUST_DEBUG -G -g


USEGLLIB	     := 1
USEPARAMGL	     := 1
USEGLUT		     := 1
USERENDERCHECKGL     := 1
USENEWINTEROP        := 1

################################################################################
# Rules and targets

include ../../common/common.mk
