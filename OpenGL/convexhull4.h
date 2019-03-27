#pragma once

#include <cstdio>

#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "point.h"
#include "utility.h"

using std::vector;

void ConvexHull4(vector<Point>& points, vector<Point>& hull);