#pragma once

#include "point.h"

// ���ص�p������߶�p1p2��λ��
// p��p1p2�Ϸ����� 1
// p��p1p2�·�����-1
// p��p1p2���߷��� 0
int FindSide(const Point& p1, const Point& p2, const Point& p);

void ConvexHull3(vector<Point> points, vector<Point>& hull);