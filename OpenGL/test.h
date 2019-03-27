#pragma once

#include <iostream>
#include <vector>

namespace Test
{
	using std::vector;

	template <typename T>
	void PrintVector(const vector<T>& vec)
	{
		for (size_t i = 0; i < vec.size(); i++) {
			std::cout << vec[i] << ' ';
		}
		printf("\n");
	}

	void Test01();
}