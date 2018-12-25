#include "utility.h"

#include <random>

float rand(float min, float max)
{
	std::mt19937 rng;
	rng.seed(std::random_device()());
	// std::uniform_real_distribution<float> distribution(min, max);
	std::normal_distribution<float> distribution(min, max);
	return distribution(rng);
}

bool is_pow_of_2(int x)
{
	return x > 1 && (x & (x - 1)) == 0;
}