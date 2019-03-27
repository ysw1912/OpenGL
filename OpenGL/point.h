#pragma once

#include <vector>

#include "gpu_helper.h"
#include "ply_parser.h"
#include "utility.h"

#define LEFT	-1
#define RIGHT	1

using std::vector;
using Point = float2;

struct PointCmp
{
	bool operator()(const Point& lhs, const Point& rhs) const;
};

void PrintPoint(const vector<Point>&, bool show_all = false);

enum RandMode { SQUARE, CIRCLE, NORMAL };
void InitPoint(vector<Point>&, size_t, RandMode);

bool ReadPlyFile(const char* filename, vector<Point>& points);

/** p��q��r�ķ���
 * �� LEFT -1 ��ʱ��(��ת)
 * �� 	    0 ����
 * �� RIGHT 1 ˳ʱ��(��ת) 
 * or say,
 ** r���߶�pq��λ��
 * �� LEFT -1 r��pq�Ϸ�/���
 * ��       0 r��pq��
 * �� RIGHT 1 r��pq�·�/�Ҳ�
 */
__host__ __device__
int16_t Orientation(const Point& p, const Point& q, const Point& r);

// ����������
void Swap(Point &lhs, Point &rhs);

// ����������ľ����ƽ��
float DistSq(const Point& lhs, const Point& rhs);

// ���㼯�����һ��
void Normalization(vector<Point>& points);

// ��p���߶�p1p2����ĳɱ���ֵ
float DistLine(const Point& p1, const Point& p2, const Point& p);

// ������pqr�����
float Area(const Point& p, const Point& q, const Point& r);