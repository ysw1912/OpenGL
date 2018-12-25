#include "point.h"
#include "convexhull3.h"

#include <set>

using std::set;

// 返回点p相对于线段p1p2的位置
// p在p1p2上方返回 1
// p在p1p2下方返回-1
// p与p1p2共线返回 0
int FindSide(const Point& p1, const Point& p2, const Point& p)
{
	return Orientation(p2, p1, p);
}

// End points of line L are p1 and p2.  side can have value 
// 1 or -1 specifying each of the parts made by the line L 
void QuickHull(const vector<Point>& points, const Point& p1, const Point& p2, int side,
	set<Point, PointCmp>& hull)
{
	int idx = -1;
	float max = 0.0f;

	// 找到位于p1p2某一边，且距离其最远的点
	for (int i = 0; i < static_cast<int>(points.size()); i++) {
		float temp = Area(p1, p2, points[i]);
		if (FindSide(p1, p2, points[i]) == side && temp > max) {
			idx = i;
			max = temp;
		}
	}
	// 若未找到点，则将p1、p2加入凸包
	if (idx == -1) {
		hull.insert(p1);
		hull.insert(p2);
		return;
	}

	// 分治递归
	// 方向应该是p2相对于线段points[idx]-p1的另一侧
	QuickHull(points, points[idx], p1, -FindSide(points[idx], p1, p2), hull);
	QuickHull(points, points[idx], p2, -FindSide(points[idx], p2, p1), hull);
}

void ConvexHull3(vector<Point> points, vector<Point>& hull)
{
	size_t n = points.size();
	if (n < 3)	return;

	extern size_t xmin, xmax;;

	// 分治
	set<Point, PointCmp> hullset;
	QuickHull(points, points[xmin], points[xmax], 1, hullset);
	hull.resize(hullset.size());
	size_t i = 0;
	for (auto it = hullset.begin(); it != hullset.end(); it++) {
		hull[i++] = *it;
	}
	hullset.clear();
	QuickHull(points, points[xmin], points[xmax], -1, hullset);
	i += hullset.size();
	hull.resize(i);
	for (auto it = hullset.begin(); it != hullset.end(); it++) {
		hull[--i] = *it;
	}
}