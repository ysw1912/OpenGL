#include "point.h"
#include "convexhull3.h"

#include <set>

using std::set;

// ���ص�p������߶�p1p2��λ��
// p��p1p2�Ϸ����� 1
// p��p1p2�·�����-1
// p��p1p2���߷��� 0
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

	// �ҵ�λ��p1p2ĳһ�ߣ��Ҿ�������Զ�ĵ�
	for (int i = 0; i < static_cast<int>(points.size()); i++) {
		float temp = Area(p1, p2, points[i]);
		if (FindSide(p1, p2, points[i]) == side && temp > max) {
			idx = i;
			max = temp;
		}
	}
	// ��δ�ҵ��㣬��p1��p2����͹��
	if (idx == -1) {
		hull.insert(p1);
		hull.insert(p2);
		return;
	}

	// ���εݹ�
	// ����Ӧ����p2������߶�points[idx]-p1����һ��
	QuickHull(points, points[idx], p1, -FindSide(points[idx], p1, p2), hull);
	QuickHull(points, points[idx], p2, -FindSide(points[idx], p2, p1), hull);
}

void ConvexHull3(vector<Point> points, vector<Point>& hull)
{
	size_t n = points.size();
	if (n < 3)	return;

	extern size_t xmin, xmax;;

	// ����
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