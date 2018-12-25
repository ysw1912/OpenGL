#include "point.h"
#include "convexhull2.h"

// ����Compare��ȫ�ֱ���
Point p0;

// ����ȽϺ������Ƚ���ȫ�ֱ���p0�ĽǶ�
// p0p1�Ƕȣ�p0p2�Ƕȣ�����-1
// p0p1�Ƕȣ�p0p2�Ƕȣ����� 1
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

// ����ջ������һ��Point 
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

	// �ҵ����·��ĵ㣬�������
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
	// ���任����һ��λ��
	Swap(points[0], points[bottommost]);

	// ��Compare����
	p0 = points[0];
	qsort(&points[1], n - 1, sizeof(Point), Compare);

	// ���ж������p0�Ƕ���ͬ�������¾�p0��Զ���Ǹ�
	int num = 1;	// ��¼ɾ��Ԫ�غ�points��Ԫ������
	for (size_t i = 1; i < n; ++i) {
		// ��i��i + 1�Ƕ���ͬ����һֱ�Ƴ�i
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
		// ��ջ���ڶ����㡢ջ���㡢points[i]�ķ�������ʱ��
		// ��һֱ�Ƴ�ջ����
		while (Orientation(NextToTop(S), S.top(), points[i]) >= 0)
			S.pop();
		S.push(points[i]);
	}

	// ջS��Ԫ�ؼ�Ϊ���
	hull.resize(S.size());
	int i = 0;
	while (!S.empty()) {
		hull[i++] = S.top();
		S.pop();
	}
}