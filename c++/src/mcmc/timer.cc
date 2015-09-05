#include "mcmc/timer.h"

#include <iomanip>

namespace mcmc {
namespace timer {

Timer::Timer(const std::string &name)
  : name(name), t_total(std::chrono::duration<int>(0)), N(0) {}

void Timer::reset() {
  t_total = std::chrono::duration<int>(0);
  N = 0;
}

void Timer::setTabular(bool on) { tabular = on; }

void Timer::printHeader(std::ostream &s) {
  if (tabular) {
    s << std::left << std::setw(nameWidth + 1) << "timer";
    s << std::right << std::setw(totalWidth + 1) << "total (s)";
    s << std::right << std::setw(tickWidth + 1) << "ticks";
    s << std::right << std::setw(perTickWidth + 1) << "per tick (us)";
    s << std::right << std::endl;
  }
}

std::ostream &Timer::put(std::ostream &s) const {
  using namespace std::chrono;

  s << std::setw(nameWidth) << std::left << name;
  if (N == 0) {
    s << "<unused>";
  } else {
    if (tabular) {
      s << " " << std::setprecision(3) << std::right << std::setw(totalWidth)
        << (duration_cast<milliseconds>(t_total).count() / (double)MILLI);
      s << " " << std::right << std::setw(tickWidth) << N;
      s << " " << std::setprecision(3) << std::fixed << std::right
        << std::setw(perTickWidth)
        << (duration_cast<nanoseconds>(t_total).count() / (double)MILLI / N);
    } else {
      s << " total " << std::setprecision(3) << std::right
        << std::setw(totalWidth)
        << (duration_cast<milliseconds>(t_total).count() / (double)MILLI)
        << "s";
      s << "; ticks " << std::right << std::setw(tickWidth) << N;
      s << "; per tick " << std::fixed << std::setprecision(3) << std::right
        << std::setw(perTickWidth)
        << (duration_cast<nanoseconds>(t_total).count() / (double)MILLI / N)
        << "us";
    }
  }

  return s;
}

std::ostream &operator<<(std::ostream &s, const Timer &timer) {
  return timer.put(s);
}

bool Timer::tabular = false;

}  // namespace timer
}  // namespace mcmc
