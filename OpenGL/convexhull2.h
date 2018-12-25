#pragma once

#include <stack>
#include <vector>

#include "point.h"

using std::stack;
using std::vector;

int Compare(const void* lhs, const void* rhs);

// 返回栈顶的下一个Point 
Point NextToTop(stack<Point> &S);

void ConvexHull2(vector<Point> points, vector<Point>& hull);