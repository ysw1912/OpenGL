#pragma once

#include "point.h"
#include "profiler.h"

#include <cmath>
#include <Windows.h>

const double PI = 4 * (atan(0.5F) + atan(0.2F) + atan(0.125F));

float uniform_rand(float min, float max);
float normal_rand(float min, float max);

// x是否是2的n次幂
bool is_pow_of_2(int x);

bool WriteBitmapFile(char* filename, int width, int height, UCHAR* bitmapData);