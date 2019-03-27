#include "point.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#include <cstdio>
#include <cstring>

using std::max;

//const float EPSILON = 0.00001F;

size_t xmin, xmax, ymin, ymax;

bool PointCmp::operator()(const Point& lhs, const Point& rhs) const
{
	return lhs.x < rhs.x || (rhs.x == lhs.x && lhs.y < rhs.y);
}

void PrintPoint(const vector<Point>& points, bool show_all)
{
	size_t n = points.size();
	if (show_all || n <= 12) {
		for (size_t i = 0; i < points.size(); i++)
			printf("(%.2f, %.2f) ", points[i].x, points[i].y);
	}
	else {
		for (size_t i = 0; i < 6; i++)
			printf("(%.2f, %.2f) ", points[i].x, points[i].y);
		printf("... ");
		for (size_t i = points.size() - 4; i < points.size(); i++)
			printf("(%.2f, %.2f) ", points[i].x, points[i].y);
	}
	printf("\n");
}

void InitPoint(vector<Point>& points, size_t size, RandMode mode)
{
	points.reserve(size);
	switch (mode)
	{
		case SQUARE: {
			for (size_t i = 0; i < size; i++) {
				points[i].x = uniform_rand(-500.0F, 500.0F);
				points[i].y = uniform_rand(-500.0F, 500.0F);
			}
			break;
		}
		case CIRCLE: {
			for (size_t i = 0; i < size; i++) {
				float theta = uniform_rand(0, 360.0F), k = uniform_rand(0, 1.0F);
				float r = 500.0F * sqrt(k);
				points[i].x = static_cast<float>(r * sin(theta * PI / 180));
				points[i].y = static_cast<float>(r * cos(theta * PI / 180));
			}
			break;
		}
		case NORMAL: {
			for (size_t i = 0; i < size; i++) {
				points[i].x = normal_rand(-500.0F, 500.0F);
				points[i].y = normal_rand(-500.0F, 500.0F);
			}
			break;
		}
	}
}

bool ReadPlyFile(const char* filename, vector<Point>& points)
{
	/*
	if (strstr(filename, ".ply") == NULL) {
		fprintf(stderr, "File must be .ply format!\n");
		return false;
	}
	FILE* file = fopen(filename, "r");
	if (!file) {
		fprintf(stderr, "Cannot open PLY file %s!\n", filename);
		return false;
	}
	printf("Reading %s...\n", filename);

	char buffer[256];
	// Read header
	if (!fgets(buffer, 256, file) || strncmp(buffer, "ply", 3)) {
		fprintf(stderr, "%s file, not a PLY file!\n", buffer);
		return false;
	}

#define GET_LINE() if (!fgets(buffer, 256, file)) goto READ_ERROR
#define LINE_IS(text) !strnicmp(buffer, text, strlen(text))

	bool isAscii;	// ply文件是ascii格式或binary格式
	GET_LINE();
	if (LINE_IS("format ascii 1.0"))
		isAscii = true;
	else if (LINE_IS("format binary_big_endian 1.0") || LINE_IS("format binary_little_endian 1.0"))
		isAscii = false;
	else
		fprintf(stderr, "Unknown PLY file format!\n");
	while (true) {
		GET_LINE();
		if (!LINE_IS("obj_info") && !LINE_IS("comment"))
			break;
	}

	size_t totalNum;
	int result = sscanf(buffer, "element vertex %zd\n", &totalNum);
	if (result != 1) {
		fprintf(stderr, "Expected \"element vertex\"！\n");
		goto READ_ERROR;
	}
	printf("  ... %zd points", totalNum);
	points.resize(totalNum);

	if (isAscii) {
		while (true) {
			GET_LINE();
			if (LINE_IS("end_header"))
				break;
		}
		// 读取点
		for (size_t i = 0; i < totalNum; ++i) {
			GET_LINE();
			sscanf(buffer, "%f %f", &points[i].x, &points[i].y);
		}
	}
	else {
		// blog.csdn.net/my_lord_/article/details/54947939
		// github.com/ddiakopoulos/ply_parser/blob/master/source/ply_parser.h
		GET_LINE();
		if (!LINE_IS("property float x")) {
			fprintf(stderr, "Expected \"property float x\"!\n");
			goto READ_ERROR;
		}
		GET_LINE();
		if (!LINE_IS("property float y")) {
			fprintf(stderr, "Expected \"property float y\"!\n");
			goto READ_ERROR;
		}
		GET_LINE();
		if (!LINE_IS("property float z")) {
			fprintf(stderr, "Expected \"property float z\"!\n");
			goto READ_ERROR;
		}

		int otherPropLen = 0;
		GET_LINE();
		while (LINE_IS("property")) {
			if (LINE_IS("property char") || LINE_IS("property uchar"))
				otherPropLen += 1;
			else if (LINE_IS("property int") || LINE_IS("property uint") || LINE_IS("property float"))
				otherPropLen += 4;
			else {
				fprintf(stderr, "Unsupported vertex property: %s\n", buffer);
				goto READ_ERROR;
			}
			GET_LINE();
		}

		while (true) {
			GET_LINE();
			if (LINE_IS("end_header"))
				break;
		}

		fflush(stdout);
		printf("\notherPropLen = %d\nReading data...\n", otherPropLen);
		for (size_t i = 0; i < totalNum; i++) {
			if (!fread((void*)&(points[i].x), 4, 1, file))
				goto READ_ERROR;
			if (!fread((void*)&(points[i].y), 4, 1, file))
				goto READ_ERROR;
			if (!fread((void*)buffer, 4, 1, file))
				goto READ_ERROR;
			if (otherPropLen && !fread((void*)buffer, otherPropLen, 1, file))
				goto READ_ERROR;
			printf("(%f, %f)\n", points[i].x, points[i].y);
		}
	}

	fclose(file);
	printf(" Loaded!\n");
	return true;

READ_ERROR:
	fclose(file);
	fprintf(stderr, "Error in reading PLY file!\n");
	return false;
	*/

	using namespace ply_parser;

	try
	{
		std::string f(filename);
		std::ifstream ss(f, std::ios::binary);
		if (ss.fail())
			throw std::runtime_error("Failed to open " + f);

		PlyFile file;
		file.parse_header(ss);
		for (auto c : file.get_comments())
			std::cout << "Comment: " << c << std::endl;
		for (auto e : file.get_elements()) {
			std::cout << "element - " << e.name << " (" << e.size << ")" << std::endl;
			for (auto p : e.properties) std::cout << "\tproperty - " << p.name << " (" << ply_parser::PropertyTable[p.propertyType].str << ")" << std::endl;
		}
		printf("----------------------------------------------\n");

		std::shared_ptr<PlyData> vertices;
		try {
			vertices = file.request_properties_from_element("vertex", {"x", "y"});
		}
		catch (const std::exception& e) {
			std::cerr << "ply_parser exception: " << e.what() << std::endl;
		}

		file.read(ss);

		if (vertices)
			std::cout << "Read " << vertices->count << " total vertices" << std::endl;
		const size_t numVerticesBytes = vertices->buffer.size_bytes();
		points.resize(vertices->count);
		std::memcpy(points.data(), vertices->buffer.get(), numVerticesBytes);
		PrintPoint(points);

		return true;
	}
	catch (const std::exception & e)
	{
		std::cerr << "Caught ply_parser exception: " << e.what() << std::endl;
		return false;
	}
}

/** p、q、r的方向
 * — LEFT -1 逆时针(左转)
 * — 	    0 共线
 * — RIGHT 1 顺时针(右转)
 * or say,
 ** r在线段pq的位置
 * — LEFT -1 r在pq上方/左侧
 * —       0 r在pq上
 * — RIGHT 1 r在pq下方/右侧
 */
__host__ __device__
int16_t Orientation(const Point& p, const Point& q, const Point& r)
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