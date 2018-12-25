#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define checkCudaError(err) __CheckCudaError( err, __FILE__, __LINE__ )
static void __CheckCudaError(cudaError_t err, const char *file, const int32_t line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(-1);
	}
}