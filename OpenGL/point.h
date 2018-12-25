#pragma once

#include <vector>

#include "gpu_helper.h"

#define LEFT	-1
#define RIGHT	1

using std::vector;
using Point = float2;

struct PointCmp
{
	bool operator()(const Point& lhs, const Point& rhs) const;
};

void PrintPoint(const vector<Point>&);

/* p->q->r的方向
* LEFT	-1 逆时针
*		 0 共线
* RIGHT 1 顺时针 */
__host__ __device__
int Orientation(const Point& p, const Point& q, const Point& r);

// 交换两个点
void Swap(Point &lhs, Point &rhs);

// 返回两个点的距离的平方
float DistSq(const Point& lhs, const Point& rhs);

// 将点集坐标归一化
void Normalization(vector<Point>& points);

// 与p与线段p1p2距离的成比例值
float DistLine(const Point& p1, const Point& p2, const Point& p);

// 三角形pqr的面积
float Area(const Point& p, const Point& q, const Point& r);