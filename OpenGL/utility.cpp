#include "utility.h"

#include <random>

float uniform_rand(float min, float max)
{
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_real_distribution<float> distribution(min, max);
	return distribution(rng);
}

float normal_rand(float min, float max)
{
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::normal_distribution<float> distribution(min, max);
	return distribution(rng);
}

bool is_pow_of_2(int x)
{
	return x > 1 && (x & (x - 1)) == 0;
}

bool WriteBitmapFile(char* filename, int width, int height, UCHAR* bitmapData)
{
	// 填充BITMAPFILEHEADER
	BITMAPFILEHEADER bitmapFileHeader;
	memset(&bitmapFileHeader, 0, sizeof(BITMAPFILEHEADER));
	bitmapFileHeader.bfSize = sizeof(BITMAPFILEHEADER);
	bitmapFileHeader.bfType = 0x4d42;	// BM
	bitmapFileHeader.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

	//填充BITMAPINFOHEADER
	BITMAPINFOHEADER bitmapInfoHeader;
	memset(&bitmapInfoHeader, 0, sizeof(BITMAPINFOHEADER));
	bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitmapInfoHeader.biWidth = width;
	bitmapInfoHeader.biHeight = height;
	bitmapInfoHeader.biPlanes = 1;
	bitmapInfoHeader.biBitCount = 24;
	bitmapInfoHeader.biCompression = BI_RGB;
	bitmapInfoHeader.biSizeImage = width * abs(height) * 3;

	FILE* filePtr;			// 连接要保存的bitmap文件用
	UCHAR tempRGB;	// 临时色素
	uint32_t imageIdx;

	//交换R、B的像素位置,bitmap的文件放置的是BGR,内存的是RGB
	for (imageIdx = 0; imageIdx < bitmapInfoHeader.biSizeImage; imageIdx += 3) {
		tempRGB = bitmapData[imageIdx];
		bitmapData[imageIdx] = bitmapData[imageIdx + 2];
		bitmapData[imageIdx + 2] = tempRGB;
	}

	if (fopen_s(&filePtr, filename, "wb") != 0)
		return false;

	fwrite(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);
	fwrite(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);
	fwrite(bitmapData, bitmapInfoHeader.biSizeImage, 1, filePtr);

	fclose(filePtr);
	return true;
}