#include <unistd.h>

#include <mr/timer.h>

#include <cstring>	// legacy strncmp

#include <string>
#include <fstream>
#include <sstream>


namespace mr {
namespace timer {

double Timer::CPU_speed_in_MHz = Timer::get_CPU_speed_in_MHz();
bool Timer::tabular = false;


double Timer::get_CPU_speed_in_MHz()
{
#if 0
#if defined __linux__
    std::ifstream infile("/proc/cpuinfo");
    char     buffer[256], *colon;

    while (infile.good()) {
		infile.getline(buffer, 256);

		if (strncmp("cpu MHz", buffer, 7) == 0 && (colon = strchr(buffer, ':')) != 0) {
#ifndef HOST_NAME_MAX
#  define HOST_NAME_MAX	256
#endif
			char host[HOST_NAME_MAX];
			(void)gethostname(host, HOST_NAME_MAX);
			std::cerr << host << ": CPU speed " << atof(colon + 2) << "MHz" << std::endl;
			return atof(colon + 2);
		}
	}

    return 0.0;
#endif
#else
	int64_t	t_start;
	int64_t	t_stop;

	rdtsc(&t_start);
	usleep(5000);
	rdtsc(&t_stop);
	double dt = (t_stop - t_start) / 5000.0;
	std::cerr << "CPU speed " << dt << "MHz" << std::endl;
	return dt;
#endif
}


void Timer::print_time(std::ostream &s, const char *which, double time) const
{
    static const char *units[] = { " ns", " us", " ms", "  s", " Ks", 0 };
    const char	      **unit   = units;

    time = 1000.0 * time / CPU_speed_in_MHz;

	while (time >= 999.5 && unit[1] != 0) {
		time /= 1000.0;
		++ unit;
	}

    s << which << std::setw(4) << std::setprecision(3) << time << *unit;
}


std::ostream &Timer::print(std::ostream &s) const
{
	if (Timer::tabular) {
		s << std::setw(nameWidth) << std::left << name;
	} else {
		s << std::left << std::setw(25) << (name.size() != 0 ? name : "timer") << ": " << std::right;
	}

	if (CPU_speed_in_MHz == 0) {
		s << "could not determine CPU speed\n";
	} else if (count > 0) {
		double total = static_cast<double>(total_time);

		if (Timer::tabular) {
			double us = total / CPU_speed_in_MHz;

			std::ios_base::fmtflags flags = s.flags();
            s << std::fixed;
			s << std::setprecision(3) << std::right;
			s << std::setprecision(3) << std::setw(totalWidth) << (us / 1000000.0);
			s << std::setw(tickWidth) << count;
			s << std::setprecision(3) << std::right << std::setw(perTickWidth) << (us / count);
            us = max / CPU_speed_in_MHz;
			s << std::setprecision(3) << std::right << std::setw(perTickWidth) << us;
			s.flags(flags);

		} else if (compact) {
			print_time(s, "  ", total / static_cast<double>(count));
			print_time(s, "  ", total);
			s << " " << std::setw(9) << count;
		} else {
			print_time(s, "avg = ", total / static_cast<double>(count));
			print_time(s, ", total = ", total);
			s << ", count = " << std::setw(9) << count;
		}

	} else {
		s << "<unused>";
	}

	return s;
}


std::string Timer::print_banner() {
	std::ostringstream o;
	o << std::left << std::setw(30) << "=== Timer ===" <<
	   	std::setw(9) << "avg" <<
		std::setw(9) << "total" <<
		std::setw(11) << "count";

	return o.str();
}


std::ostream &operator << (std::ostream &str, const Timer &timer)
{
    return timer.print(str);
}

double Timer::getTimeInSeconds() const
{
    double total = static_cast<double>(total_time);
    double res = (total / 1000000.0) / CPU_speed_in_MHz;
    return res;
}

}   // namespace Timer
}   // namespace mr
