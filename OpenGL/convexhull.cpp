//#include <glad/glad.h>
#include <GLFW/glfw3.h>  
#include <iostream>
#include <set>

#include <libqhullcpp/RboxPoints.h>
#include "libqhullcpp/PointCoordinates.h"
#include <libqhullcpp/Qhull.h>
#include <libqhullcpp/QhullError.h>
#include <libqhullcpp/QhullFacet.h>
#include <libqhullcpp/QhullFacetList.h>
#include <libqhullcpp/QhullLinkedList.h>
#include <libqhullcpp/QhullQh.h>
#include <libqhullcpp/QhullVertex.h>
#include <libqhullcpp/QhullVertexSet.h>
#include <libqhullcpp/QhullSet.h>

#include "convexhull.h"
#include "convexhull1.h"
#include "convexhull2.h"
#include "convexhull3.h"
#include "convexhull4.h"
#include "point.h"

void QHull(const vector<Point>& points, vector<Point>& hull)
{
	QHULL_LIB_CHECK

	orgQhull::Qhull qhull;

	//orgQhull::PointCoordinates pc(qhull);
	orgQhull::PointCoordinates pc(qhull, 2, "PointCoordinates");
	printf("%d-d, %d-size, \"%s\"\n", pc.dimension(), pc.count(), pc.comment().c_str());

	vector<double> vec{ 0.0, 0.0, 2.0, 0.0, 0.0, 2.0 };
	pc.append(vec);
	printf("%d-d, %d-size, \"%s\"\n", pc.dimension(), pc.count(), pc.comment().c_str());

	qhull.runQhull("", 2, 3, vec.data(), "");
	printf("hullDimension = %d\n", qhull.hullDimension());
	printf("area = %f\n", qhull.area());
	qhull.outputQhull();

	//QCOMPARE(pc.size(), 0U);

	/*
	vector<vec3> vertices;
	qhull.runQhull3D(vertices, "Qt");

	QhullFacetList facets = qhull.facetList();
	for (QhullFacetList::iterator it = facets.begin(); it != facets.end(); ++it)
	{
		if (!(*it).isGood()) continue;
		QhullFacet f = *it;
		QhullVertexSet vSet = f.vertices();
		for (QhullVertexSet::iterator vIt = vSet.begin(); vIt != vSet.end(); ++vIt)
		{
			QhullVertex v = *vIt;
			QhullPoint p = v.point();
			double * coords = p.coordinates();
			vec3 aPoint = vec3(coords[0], coords[1], coords[2]);
			// ...Do what ever you want
		}
	}

	// Another way to iterate (c++11), and the way the get the normals
	std::vector<std::pair<vec3, double> > facetsNormals;
	for each (QhullFacet facet in qhull.facetList().toStdVector())
	{
		if (facet.hyperplane().isDefined())
		{
			auto coord = facet.hyperplane().coordinates();
			vec3 normal(coord[0], coord[1], coord[2]);
			double offset = facet.hyperplane().offset();
			facetsNormals.push_back(std::pair<vec3, double>(normal, offset));
		}
	}
	*/
}

void error_callback(int error, const char* description)
{
	fputs(description, stderr);
}

void close_callback(GLFWwindow* window)
{
	fputs("关闭窗口\n", stderr);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void process_input(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

void SaveScreenShot(int width, int height)
{
	int len = width * height * 3;
	void* screenData = malloc(len);
	memset(screenData, 0, len);

	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, screenData);

	WriteBitmapFile("F:\\points\\screenshot.bmp", width, height, (UCHAR*)screenData);

	free(screenData);
}

void TestConvexHull()
{
	vector<Point> points;
	char filename[75] = "F:\\points\\";
	char input[64];
	scanf("%s", input);
	strncat(filename, input, 64);
	ReadPlyFile(filename, points);

	// SQUARE, CIRCLE, NORMAL
	//InitPoint(points, N, CIRCLE);
	// printf("points: "); PrintPoint(points);

	vector<Point> hull;
	printf("凸包求解...\n");

	printf("点集标准化...去重...\n");
	Normalization(points);
	std::set<Point, PointCmp> st(points.begin(), points.end());
	points.assign(st.begin(), st.end());

	Profiler::Start();
	ConvexHull4(points, hull);
	Profiler::Finish();
	
	printf("ConvexHull: %f (ms)\n", Profiler::dumpDuration());
	printf("hull:   "); PrintPoint(hull);

	/******************** 图像展示 ********************/
	if (!glfwInit())
		exit(-1);

	glfwSetErrorCallback(error_callback);

	int width = 640, height = 640;
	GLFWwindow* window = glfwCreateWindow(width, height, "Hello World", NULL, NULL);
	if (!window) {
		glfwTerminate();
		exit(-1);
	}

	glfwMakeContextCurrent(window);
	glfwSetWindowCloseCallback(window, close_callback);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);


	/*
	// GLAD用来管理OpenGL的函数指针，在调用任何OpenGL函数之前需要初始化GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		glfwTerminate();
		exit(-1);
	}
	glViewport(0, 0, 800, 600);
	*/

	int count = 0;
	// 渲染循环
	while (!glfwWindowShouldClose(window)) {
		// 输入
		process_input(window);

		// 渲染指令
		//glClearColor(0.2f, 0.3f, 0.3f, 1.0f);	// 绿底
		glClearColor(1.0F, 1.0F, 1.0F, 1.0F);	// 白底
		glClear(GL_COLOR_BUFFER_BIT);
		
		glPointSize(4);
		glBegin(GL_POINTS);
		// 画所有点
		glColor3f(0.7F, 0.7F, 0.7F);
		for (size_t i = 0; i < points.size(); i++) {
			glVertex2f(points[i].x, points[i].y);
		}
		// 画hull点
		//glColor3f(1.0, 0.0, 0.0);
		//for (size_t i = 0; i < hull.size(); i++) {
		//	glVertex2f(hull[i].x, hull[i].y);
		//}
		glEnd();

		//画凸包hull 
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glColor3f(0.0F, 0.0F, 0.0F);
		glLineWidth(4.0F);	// 0-10.0F
		glBegin(GL_POLYGON);
		for (size_t i = 0; i < hull.size(); i++) {
			glVertex2f(hull[i].x, hull[i].y);
		}
		glEnd();

		if (++count == 10)
			SaveScreenShot(width, height);

		// 交换缓冲
		glfwSwapBuffers(window);
		// 检查并调用事件
		glfwPollEvents();
	}

	glfwTerminate();
}