#pragma once

#include <cstdio>

#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "point.h"
#include "utility.h"

using std::vector;

__device__ void make_remote(Point* p);
void show_current_hoods(FILE* out, Point* h_hood, int count, int d);
__device__ short g(Point *d_hood, short i, short j, short start, short d);
__device__ short f(Point *d_hood, short i, short j, short start, short d);
__global__ void match_and_merge(Point *d_hood, Point *d_newhood, short *d_scratch);

void ConvexHull4(vector<Point> points, vector<Point>& hull);