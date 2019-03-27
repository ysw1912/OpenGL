#include "point.h"
#include "convexhull3.h"

#include <set>

using std::set;

// �߶�p1p2��ĳһ��side
void QuickHull(const vector<Point>& points,
			   const Point& p1, const Point& p2, int16_t side,
			   set<Point, PointCmp>& hull)
{
	int idx = -1;
	float max = -1.0F;

	// �ҵ�λ��p1p2ĳһ�ߣ��Ҿ�������Զ�ĵ�
	for (int i = 0; i < static_cast<int>(points.size()); i++) {
		float temp = Area(p1, p2, points[i]);
		if (Orientation(p1, p2, points[i]) == side && temp > max) {
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
	QuickHull(points, points[idx], p1, -Orientation(points[idx], p1, p2), hull);
	// ����Ӧ����p1������߶�points[idx]-p2����һ��
	QuickHull(points, points[idx], p2, -Orientation(points[idx], p2, p1), hull);
}

void ConvexHull3(vector<Point> points, vector<Point>& hull)
{
	size_t n = points.size();
	if (n < 3)	return;

	extern size_t xmin, xmax;

	// ����
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