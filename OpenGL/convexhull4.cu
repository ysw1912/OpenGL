#include "convexhull4.h"

const Point REMOTE = { 10.0F, 0.0F };

__device__ void make_remote(Point* p)
{
	p->x = 10.0f;
	p->y = 0.0f;
}

void show_current_hoods(FILE* out, Point* h_hood, int count, int d)
{
	int hoodsize;
	fprintf(out, "%d\n", count / d);
	for (int i = 0; i < count / d; ++i) {
		hoodsize = 0;
		for (int j = 0; j < d; ++j)
			if (h_hood[i * d + j].x <= 1.0)
				++hoodsize;
		fprintf(out, "%d\n", hoodsize);
	}
	fprintf(out, "\n");
}

__device__ short g(Point *d_hood, short i, short j, short start, short d)
{
	Point p, q, q_next, q_prev;
	int atstart, atend;
	int isleft;
	if (d_hood[j].x > 1) /* REMOTE */
		return HIGH;
	p = d_hood[i];
	q = d_hood[j];
	atend = (j == start + 2 * d - 1 || d_hood[j + 1].x > 1.0);
	q_next = d_hood[j + 1 - atend];
	q_next.y -= (float)atend;
	if (Orientation(p, q, q_next) == -1)
		return LOW;
	atstart = (j == start + d);
	q_prev = d_hood[j + atstart - 1];
	q_prev.y -= (float)atstart;
	if (Orientation(p, q, q_prev) == -1)
		isleft = 1;
	else
		isleft = 0;
	return HIGH * isleft + EQUAL * (1 - isleft);
}

__device__ short f(Point *d_hood, short i, short j, short start, short d)
{
	Point p, q, p_next, p_prev;
	int atstart, atend;
	int isleft;
	if (d_hood[i].x > 1) /* REMOTE */
		return HIGH;
	p = d_hood[i];
	q = d_hood[j];
	atend = (i == start + d - 1 || d_hood[i + 1].x > 1);
	p_next = d_hood[i + 1 - atend];
	p_next.y -= (float)atend;
	if (Orientation(p, q, p_next) == -1)
		return LOW;
	atstart = (i == start);
	p_prev = d_hood[i + atstart - 1];
	p_prev.y -= (float)atstart;
	if (Orientation(p, q, p_prev) == -1)
		isleft = 1;
	else
		isleft = 0;
	return HIGH * isleft + EQUAL * (1 - isleft);
}

__global__ void match_and_merge(Point *d_hood, Point *d_newhood, short *d_scratch)
{
	uint32_t d1 = blockDim.x;
	uint32_t d2 = blockDim.y;
	uint32_t d = d1 * d2;
	uint32_t start = blockIdx.x * 2 * d;
	uint32_t x = threadIdx.x;
	uint32_t y = threadIdx.y;
	uint32_t idx = x + d1 * y;
	int i, j, pindex, qindex, shift;
	pindex = qindex = -1;
	d_scratch[start + idx] = -1;
	d_scratch[start + idx + d] = -1;
	__syncthreads();
	i = start + d2 * x;

	if (d_hood[i].x <= 1.0) {	/* not REMOTE */
		j = start + d + d1 * y;
		/*
		* The condition below should identify the
		* unique interval of H(Q) touching the
		* tangent from hood[i].
		*/
		if (g(d_hood, i, j, start, d) <= EQUAL &&
			(y == d2 - 1 || d_hood[j + d1].x > 1.0 || g(d_hood, i, j + d1, start, d) == HIGH))
			d_scratch[start + x] = j;
	}
	__syncthreads();
}

void ConvexHull4(vector<Point> points, vector<Point>& hull)
{
	Point *h_hood;
	short *h_scratch;
	Point *d_hood, *d_newhood;
	short *d_scratch;

	size_t count = points.size();
	if (!is_pow_of_2(count)) {
		fprintf(stderr, "Count %zd not a power of 2, abort\n", count);
		exit(-1);
	}
	printf("count: %zd\n", count);

	h_hood = (Point*)malloc(count * sizeof(Point));
	h_scratch = (short*)malloc(count * sizeof(short));
	for (size_t i = 0; i < count; ++i) {
		h_hood[i] = points[i];
	}
	int d1 = 2;
	int d2 = 1;
	int d = d1 * d2;

	checkCudaError(cudaMalloc((void**)&d_hood, count * sizeof(Point)));
	checkCudaError(cudaMalloc((void**)&d_newhood, count * sizeof(Point)));
	checkCudaError(cudaMalloc((void**)&d_scratch, count * sizeof(short)));

	while (d < count) {
		show_current_hoods(stdout, h_hood, count, d);
	
		checkCudaError(cudaMemcpy(d_hood, h_hood, count * sizeof(Point), cudaMemcpyHostToDevice));
	
		dim3 range(count / (2 * d));
		dim3 block(d1, d2);
		match_and_merge<<<range, block>>>(d_hood, d_newhood, d_scratch);

		checkCudaError(cudaMemcpy(h_hood, d_newhood, count * sizeof(Point), cudaMemcpyDeviceToHost));
		checkCudaError(cudaMemcpy(h_scratch, d_scratch, count * sizeof(short), cudaMemcpyDeviceToHost));
		printf("#returned from match_and_merge, d1 = %d, d2 = %d, d = %d\n", d1, d2, d);
		
		printf("#scratch:\n#");
		for (size_t i = 0; i < count; ++i) {
			printf("%3d ", h_scratch[i]);
			if (i % 10 == 0)
				printf("\n#");
		}
		printf("\n");
		if (d1 > d2)
			d2 *= 2;
		else
			d1 *= 2;
		d *= 2;
	}
#ifdef _DEBUG
	checkCudaError(cudaMemcpy(h_hood, d_newhood, count * sizeof(Point), cudaMemcpyDeviceToHost));
	printf("#newhood contents\n");
	for (size_t i = 0; i < count; ++i)
		printf("#%f %f\n", h_hood[i].x, h_hood[i].y);
	show_current_hoods(stdout, h_hood, count, d);
#endif // _DEBUG

}