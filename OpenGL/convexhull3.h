#pragma once

#include "point.h"

// 返回点p相对于线段p1p2的位置
// p在p1p2上方返回 1
// p在p1p2下方返回-1
// p与p1p2共线返回 0
int FindSide(const Point& p1, const Point& p2, const Point& p);

void ConvexHull3(vector<Point> points, vector<Point>& hull);