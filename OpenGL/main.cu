//#include <glad/glad.h>
#include <GLFW/glfw3.h>  
#include <iostream>

#include "point.h"
#include "convexhull1.h"
#include "convexhull2.h"
#include "convexhull3.h"
#include "convexhull4.h"
#include "utility.h"

void error_callback(int error, const char* description)
{
	fputs(description, stderr);
}

void close_callback(GLFWwindow* window)
{
	fputs("�رմ���\n", stderr);
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

int main(void)
{
	if (!glfwInit())
		return -1;

	glfwSetErrorCallback(error_callback);

	GLFWwindow* window = glfwCreateWindow(800, 600, "Hello World", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetWindowCloseCallback(window, close_callback);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	/*
	// GLAD��������OpenGL�ĺ���ָ�룬�ڵ����κ�OpenGL����֮ǰ��Ҫ��ʼ��GLAD
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
	glfwTerminate();
	return -1;
	}
	glViewport(0, 0, 800, 600);
	*/

	size_t N = 16;
	vector<Point> points(N);
	printf("���� %zd �������...\n", N);
	for (size_t i = 0; i < N; i++) {
		points[i].x = rand(0.0f, 800.0f);
		points[i].y = rand(0.0f, 600.0f);
	}
	printf("�㼯��׼��...\n");
	Normalization(points);
	// printf("points: "); PrintPoint(points);
	vector<Point> hull;
	printf("͹�����...\n");
	ConvexHull3(points, hull);
	// printf("hull:   "); PrintPoint(hull);

	// ��Ⱦѭ��
	while (!glfwWindowShouldClose(window)) {
		// ����
		process_input(window);

		// ��Ⱦָ��
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glColor3f(1.0, 0.0, 0.0);
		glBegin(GL_POLYGON);
		for (size_t i = 0; i < hull.size(); i++) {
			glVertex2f(hull[i].x, hull[i].y);
		}
		glEnd();

		glColor3f(1.0, 1.0, 1.0);
		glPointSize(4);
		glBegin(GL_POINTS);
		for (size_t i = 0; i < points.size(); i++) {
			glVertex2f(points[i].x, points[i].y);
		}
		glEnd();

		// ��������
		glfwSwapBuffers(window);
		// ��鲢�����¼�
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}