#ifndef __PROFILER__
#define __PROFILER__

#include <chrono>
#include <ratio>

class Profiler
{
public:
	using HRClock = std::chrono::high_resolution_clock;
	using TimePoint = HRClock::time_point;
	using SecDuration = std::chrono::duration<double, std::ratio<1, 1>>;

private:
	static SecDuration  duration;
	static TimePoint    start;
	static TimePoint    finish;

public:
	static void Start()     // 开始计时
	{
		start = HRClock::now();
	}

	static void Finish()    // 结束计时
	{
		finish = HRClock::now();
		duration = std::chrono::duration_cast<SecDuration>(finish - start);
	}

	static double dumpDuration()  // 打印时间
	{
		return duration.count() * 1000;
	}
};

#endif 