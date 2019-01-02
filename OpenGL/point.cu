#include "point.h"

#include <algorithm>
#include <cstdio>

using std::max;

size_t xmin, xmax, ymin, ymax;

bool PointCmp::operator()(const Point& lhs, const Point& rhs) const
{
	return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
}

void PrintPoint(const vector<Point>& points)
{
	for (size_t i = 0; i < points.size(); i++)
		printf("(%.2f, %.2f) ", points[i].x, points[i].y);
	printf("\n");
}

/* p->q->r的方向
 * LEFT	-1 逆时针
 *		 0 共线
 * RIGHT 1 顺时针 */
__host__ __device__
int Orientation(const Point& p, const Point& q, const Point& r)
{
	float v = (q.y - p.y) * (r.x - q.x) - (r.y - q.y) * (q.x - p.x);
	// v > 0 则 LEFT
	// v < 0 则 RIGHT
	return LEFT * (v > 0) + RIGHT * (v < 0);
}

// 交换两个点
void Swap(Point &lhs, Point &rhs)
{
	Point temp = lhs;
	lhs = rhs;
	rhs = temp;
}

// 返回两个点的距离的平方
float DistSq(const Point& lhs, const Point& rhs)
{
	return (lhs.x - rhs.x) * (lhs.x - rhs.x) + (lhs.y - rhs.y) * (lhs.y - rhs.y);
}

// 将点集坐标归一化
void Normalization(vector<Point>& points)
{
	if (points.empty())	return;
	xmin = 0, xmax = 0, ymin = 0, ymax = 0;
	for (size_t i = 1; i < points.size(); i++) {
		if (points[i].x < points[xmin].x)
			xmin = i;
		if (points[i].x > points[xmax].x)
			xmax = i;
		if (points[i].y < points[ymin].y)
			ymin = i;
		if (points[i].y > points[ymax].y)
			ymax = i;
	}
	float xmid = points[xmin].x + (points[xmax].x - points[xmin].x) / 2;
	float ymid = points[ymin].y + (points[ymax].y - points[ymin].y) / 2;
	float width = max(points[xmax].x - points[xmin].x, points[ymax].y - points[ymin].y) / 1.7f;
	for (size_t i = 0; i < points.size(); i++) {
		points[i].x -= xmid;
		points[i].x /= width;
		points[i].y -= ymid;
		points[i].y /= width;
	}
}

// 与p与线段p1p2距离的成比例值
float DistLine(const Point& p1, const Point& p2, const Point& p)
{
	return abs((p.y - p1.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p.x - p1.x));
}

// 三角形pqr的面积
float Area(const Point& p, const Point& q, const Point& r)
{
	return abs(p.x * q.y + r.x * p.y + q.x * r.y - r.x * q.y - q.x * p.y - p.x * r.y);
}