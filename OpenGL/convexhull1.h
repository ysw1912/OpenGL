#pragma once

#include <vector>

#include "point.h"

using std::vector;

// ����p��q��r���㹲�ߣ��ж�q�Ƿ����߶�pr��
bool OnSegment(const Point& p, const Point& q, const Point& r);

// �ж��߶�ab���߶�cd�Ƿ��ཻ
bool Intersect(const Point& a, const Point& b, const Point& c, const Point& d);

void ConvexHull1(const vector<Point>& points, vector<Point>& hull);