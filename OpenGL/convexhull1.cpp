#include "point.h"
#include "convexhull1.h"

#include <algorithm>

using std::max;
using std::min;

// 给定p、q、r三点共线，判断q是否在线段pr上
bool OnSegment(const Point& p, const Point& q, const Point& r)
{
	if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
		q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y))
		return true;
	return false;
}

// 判断线段ab与线段cd是否相交
bool Intersect(const Point& a, const Point& b, const Point& c, const Point& d)
{
	int o1 = Orientation(a, b, c);
	int o2 = Orientation(a, b, d);
	int o3 = Orientation(c, d, a);
	int o4 = Orientation(c, d, b);

	if (o1 != o2 && o3 != o4)
		return true;

	// a、b、c共线且c在线段ab上
	if (o1 == 0 && OnSegment(a, c, b)) return true;
	if (o2 == 0 && OnSegment(a, d, b)) return true;
	if (o3 == 0 && OnSegment(c, a, d)) return true;
	if (o4 == 0 && OnSegment(c, b, d)) return true;

	return false;
}

void ConvexHull1(const vector<Point>& points, vector<Point>& hull)
{
	size_t n = points.size();
	if (n < 3)	return;

	size_t leftmost = 0;
	for (size_t i = 1; i < n; ++i)
		if (points[i].x < points[leftmost].x)
			leftmost = i;

	size_t p = leftmost;
	do {
		hull.push_back(points[p]);

		size_t q = (p + 1) % n;
		// 找到q，相对于p是最逆时针的
		for (size_t i = 0; i < n; ++i) {
			// p->i->q方向为逆时针，因此i比q更加逆时针，更新q
			if (Orientation(points[p], points[i], points[q]) < 0)
				q = i;
		}

		p = q;
	} while (p != leftmost);
}