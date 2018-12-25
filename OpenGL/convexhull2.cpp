#include "point.h"
#include "convexhull2.h"

// 用于Compare的全局变量
Point p0;

// 排序比较函数，比较与全局变量p0的角度
// p0p1角度＜p0p2角度，返回-1
// p0p1角度＞p0p2角度，返回 1
int Compare(const void* lhs, const void* rhs)
{
	Point* p1 = (Point*)lhs;
	Point* p2 = (Point*)rhs;
	int o = Orientation(p0, *p1, *p2);
	if (o == 0)
		return (DistSq(p0, *p1) <= DistSq(p0, *p2)) ? -1 : 1;
	else
		return (o < 0) ? -1 : 1;
}

// 返回栈顶的下一个Point 
Point NextToTop(stack<Point> &S)
{
	Point p = S.top();
	S.pop();
	Point res = S.top();
	S.push(p);
	return res;
}

void ConvexHull2(vector<Point> points, vector<Point>& hull)
{
	size_t n = points.size();

	// 找到最下方的点，优先左边
	size_t bottommost = 0;
	// float ymin = points[0].y;
	extern size_t ymin;
	float min = points[ymin].y;
	for (size_t i = 1; i < n; ++i)
	{
		float y = points[i].y;
		if (min == y && points[i].x < points[bottommost].x)
			bottommost = i;
	}
	// 将其换到第一个位置
	Swap(points[0], points[bottommost]);

	// 用Compare排序
	p0 = points[0];
	qsort(&points[1], n - 1, sizeof(Point), Compare);

	// 若有多个点与p0角度相同，仅留下距p0最远的那个
	int num = 1;	// 记录删除元素后points的元素数量
	for (size_t i = 1; i < n; ++i) {
		// 当i与i + 1角度相同，则一直移除i
		while (i < n - 1 && Orientation(p0, points[i], points[i + 1]) == 0)
			i++;
		points[num] = points[i];
		num++;
	}

	if (num < 3) return;

	stack<Point> S;
	S.push(points[0]);
	S.push(points[1]);
	S.push(points[2]);

	for (int i = 4; i < num; i++) {
		// 若栈顶第二个点、栈顶点、points[i]的方向不是逆时针
		// 则一直移除栈顶点
		while (Orientation(NextToTop(S), S.top(), points[i]) >= 0)
			S.pop();
		S.push(points[i]);
	}

	// 栈S中元素即为输出
	hull.resize(S.size());
	int i = 0;
	while (!S.empty()) {
		hull[i++] = S.top();
		S.pop();
	}
}