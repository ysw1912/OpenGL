#include "convexhull4.h"

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

struct ComparePointByX
{
	__host__ __device__ bool operator()(const Point lhs, const Point rhs) const
	{
		return lhs.x < rhs.x;
	}
};

// 线段d_Points[i]d_Points[j]的某一边side
__global__ void UpdateState(Point* d_Points, uint32_t* d_State,
	uint32_t i, uint32_t j)
{
	uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	int16_t v = Orientation(d_Points[i], d_Points[j], d_Points[idx]);
	d_State[idx] = 0 * (v == -1) + 1 * (v == 1);
}

void ConvexHull4(vector<Point>& points, vector<Point>& hull)
{
	size_t size = points.size();

	Point* d_Points;
	checkCudaError(cudaMalloc((void**)&d_Points, size * sizeof(Point)));
	checkCudaError(cudaMemcpy(d_Points, points.data(), size * sizeof(Point), cudaMemcpyHostToDevice));

	// Step 1: 取x值最大和最小点的位置
	thrust::device_vector<Point> t_Points(d_Points, d_Points + size);
	thrust::device_vector<Point>::iterator it;
	it = thrust::max_element(t_Points.begin(), t_Points.end(), ComparePointByX());
	size_t maxPos = it - t_Points.begin();
	it = thrust::min_element(t_Points.begin(), t_Points.end(), ComparePointByX());
	size_t minPos = it - t_Points.begin();

	hull.push_back(points[maxPos]);
	hull.push_back(points[minPos]);

	// 创建初始的段首d_SegHead数组, 仅第1个元素为1, 其余全0
	// 创建初始的状态d_State数组
	uint32_t *h_SegHead = new uint32_t[size], *h_State = new uint32_t[size];
	memset(h_SegHead, 0, size * sizeof(uint32_t));
	memset(h_State, 0, size * sizeof(uint32_t));
	h_SegHead[0] = 1;

	uint32_t *d_SegHead, *d_State;
	checkCudaError(cudaMalloc((void**)&d_SegHead, size * sizeof(uint32_t)));
	checkCudaError(cudaMalloc((void**)&d_State, size * sizeof(uint32_t)));
	checkCudaError(cudaMemcpy(d_SegHead, h_SegHead, size * sizeof(uint32_t), cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(d_State, h_State, size * sizeof(uint32_t), cudaMemcpyHostToDevice));

	UpdateState<<<(size + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_Points, d_State, minPos, maxPos);


	// 释放内存
	delete[] h_SegHead;
	delete[] h_State;

	checkCudaError(cudaFree(d_SegHead));
	checkCudaError(cudaFree(d_State));
	checkCudaError(cudaFree(d_Points));
}