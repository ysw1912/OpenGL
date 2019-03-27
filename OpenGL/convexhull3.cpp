#include "point.h"
#include "convexhull3.h"

#include <set>

using std::set;

// 线段p1p2的某一边side
void QuickHull(const vector<Point>& points,
			   const Point& p1, const Point& p2, int16_t side,
			   set<Point, PointCmp>& hull)
{
	int idx = -1;
	float max = -1.0F;

	// 找到位于p1p2某一边，且距离其最远的点
	for (int i = 0; i < static_cast<int>(points.size()); i++) {
		float temp = Area(p1, p2, points[i]);
		if (Orientation(p1, p2, points[i]) == side && temp > max) {
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
	QuickHull(points, points[idx], p1, -Orientation(points[idx], p1, p2), hull);
	// 方向应该是p1相对于线段points[idx]-p2的另一侧
	QuickHull(points, points[idx], p2, -Orientation(points[idx], p2, p1), hull);
}

void ConvexHull3(vector<Point> points, vector<Point>& hull)
{
	size_t n = points.size();
	if (n < 3)	return;

	extern size_t xmin, xmax;

	// 分治
	set<Point, PointCmp> hullset;
	printf("Left...  ");
	QuickHull(points, points[xmin], points[xmax], LEFT, hullset);
	hull.resize(hullset.size());
	size_t i = 0;
	for (auto it = hullset.begin(); it != hullset.end(); it++) {
		hull[i++] = *it;
	}
	hullset.clear();
	printf("Right...\n");
	QuickHull(points, points[xmin], points[xmax], RIGHT, hullset);
	i += hullset.size();
	hull.resize(i);
	for (auto it = hullset.begin(); it != hullset.end(); it++) {
		hull[--i] = *it;
	}
}