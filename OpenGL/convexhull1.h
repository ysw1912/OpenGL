#pragma once

#include <vector>

#include "point.h"

using std::vector;

// 给定p、q、r三点共线，判断q是否在线段pr上
bool OnSegment(const Point& p, const Point& q, const Point& r);

// 判断线段ab与线段cd是否相交
bool Intersect(const Point& a, const Point& b, const Point& c, const Point& d);

void ConvexHull1(const vector<Point>& points, vector<Point>& hull);