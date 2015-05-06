#ifndef mr_timer_h
#define mr_timer_h

#include <stdint.h>

#include <iostream>
#include <iomanip>

#define createTimer(a) timer a(#a)

#define __PATHSCALE__

namespace mr {
namespace timer {

class Timer {
public:
	Timer(const std::string &name = std::string(""))
			: name(name),
			  write_on_exit(0),
			  compact(false) {
		reset();
	}

	Timer(const std::string &name, std::ostream &write_on_exit)
			: name(name),
			  write_on_exit(&write_on_exit),
			  compact(false) {
		reset();
	}

	~Timer() {
		if (write_on_exit != 0) {
			print(*write_on_exit);
		}
	}


	inline void start() {
#if (defined __PATHSCALE__) && (defined __i386 || defined __x86_64)
		uint32_t eax, edx;

		asm volatile ("rdtsc" : "=a" (eax), "=d" (edx));

		t_start = ((uint64_t) edx << 32) + eax;
		total_time -= t_start;
#elif (defined __GNUC__ || defined __INTEL_COMPILER) && (defined __i386 || defined __x86_64)
		asm volatile
		(
		"rdtsc\n\t"
		"subl %%eax, %0\n\t"
		"sbbl %%edx, %1"
		:
		"+m" (two_int.low), "+m" (two_int.high)
		:
		:
		"eax", "edx"
		);
#else
#error Compiler/Architecture not recognized
#endif
	}


	inline void stop() {
#if (defined __PATHSCALE__) && (defined __i386 || defined __x86_64)
		uint32_t eax, edx;

		asm volatile ("rdtsc" : "=a" (eax), "=d" (edx));

		int64_t t = ((uint64_t) edx << 32) + eax;
		total_time += t;
        if (t - t_start > max) {
          max = t - t_start;
        }
#elif (defined __GNUC__ || defined __INTEL_COMPILER) && (defined __i386 || defined __x86_64)
		asm volatile
		(
		"rdtsc\n\t"
		"addl %%eax, %0\n\t"
		"adcl %%edx, %1"
		:
		"+m" (two_int.low), "+m" (two_int.high)
		:
		:
		"eax", "edx"
		);
#endif

		++ count;
	}

	inline void reset() {
		total_time = 0;
		count      = 0;
	}

	static void setTabular(bool on) {
		tabular = on;
	}

	static void printHeader(std::ostream &s) {
		if (tabular) {
			s << std::left << std::setw(nameWidth) << "timer";
			s << std::right << std::setw(totalWidth) << "total (s)";
			s << std::right << std::setw(tickWidth) << "ticks";
			s << std::right << std::setw(perTickWidth) << "per tick (us)";
			s << std::right << std::setw(perTickWidth) << "max tick (us)";
			s << std::right << std::endl;
		}       
	}


	std::ostream &print(std::ostream &) const;

	double getTimeInSeconds() const;

	/**
	 * Alias for compatibility with mcmc::timer::Timer
	 */
	double total() const {
		return getTimeInSeconds();
	}

	static inline uint64_t now() {
		uint64_t	t;
		getTime(&t);
		return t;
	}

	static inline double t2d(uint64_t t) {
		double total = static_cast<double>(t);
		return (total / 1000000.0) / CPU_speed_in_MHz;
	}

	void setCompact(bool on = true) {
		compact = on;
	}

	static std::string print_banner();

private:
	void print_time(std::ostream &, const char *which, double time) const;

	static void	getTime(uint64_t *t) {
#if (defined __i386 || defined __x86_64)
		uint32_t eax, edx;

		asm volatile ("rdtsc" : "=a" (eax), "=d" (edx));

		*t = ((uint64_t) edx << 32) + eax;
#else
#  error Architecture not recognized
#endif
	}

	union {
		int64_t		total_time;
		struct {
#if defined __PPC__
			int		high, low;
#else
			int		low, high;
#endif
		} two_int;
	};

    int64_t         t_start;
    int64_t         max = 0;
	uint64_t	 	count;
	std::string		name;
	std::ostream   *write_on_exit;

	bool			compact;

	static double	CPU_speed_in_MHz;
	static double	get_CPU_speed_in_MHz();

	static const ::size_t nameWidth = 36;
	static const ::size_t totalWidth = 12;
	static const ::size_t tickWidth = 8;
	static const ::size_t perTickWidth = 14;

	static bool tabular;
};


std::ostream &operator << (std::ostream &, const Timer &);

#ifdef VTIMERS
#  define TIMER_START(timer) do { (timer).start(); } while (0)
#  define TIMER_STOP(timer)  do { (timer).stop(); } while (0)
#else
#  define TIMER_START(timer) do { } while (0)
#  define TIMER_STOP(timer)  do { } while (0)
#endif

}   // namespace timer
}   // namespace mr

#endif
