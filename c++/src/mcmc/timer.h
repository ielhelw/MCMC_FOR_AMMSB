#ifndef MCMC_TIMER_H__
#define MCMC_TIMER_H__

#include <iostream>
#include <iomanip>
#include <chrono>

#include "mcmc/config.h"

namespace mcmc {
namespace timer {

class Timer {
 public:
  Timer(const std::string &name = "")
      : name(name), t_total(std::chrono::duration<int>(0)), N(0) {}

  inline void start() { t_start = std::chrono::high_resolution_clock::now(); }

  inline void stop() {
    t_total += std::chrono::high_resolution_clock::now() - t_start;
    N++;
  }

  inline std::chrono::high_resolution_clock::duration total() const {
    return t_total;
  }

  inline ::size_t ticks() const { return N; }

  inline void reset() {
    t_total = std::chrono::duration<int>(0);
    N = 0;
  }

  static void setTabular(bool on) { tabular = on; }

  static void printHeader(std::ostream &s) {
    if (tabular) {
      s << std::left << std::setw(nameWidth) << "timer";
      s << std::right << std::setw(totalWidth) << "total (s)";
      s << std::right << std::setw(tickWidth) << "ticks";
      s << std::right << std::setw(perTickWidth) << "per tick (us)";
      s << std::right << std::endl;
    }
  }

  std::ostream &put(std::ostream &s) const {
    using namespace std::chrono;

    s << std::setw(nameWidth) << std::left << name;
    if (N == 0) {
      s << "<unused>";
    } else {
      if (tabular) {
        s << std::setprecision(3) << std::right << std::setw(totalWidth)
          << (duration_cast<milliseconds>(t_total).count() / (double)MILLI);
        s << std::right << std::setw(tickWidth) << N;
        s << std::setprecision(3) << std::fixed << std::right
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

 protected:
  static const ::size_t MILLI = 1000;
  static const ::size_t MICRO = 1000000;
  static const ::size_t NANO = 1000000000;
  static const ::size_t nameWidth = 36;
  static const ::size_t totalWidth = 12;
  static const ::size_t tickWidth = 8;
  static const ::size_t perTickWidth = 14;

  std::string name;
  std::chrono::high_resolution_clock::time_point t_start;
  std::chrono::high_resolution_clock::duration t_total;
  ::size_t N;
  static bool tabular;
};

inline std::ostream &operator<<(std::ostream &s, const Timer &timer) {
  return timer.put(s);
}

}  // namespace mcmc
}  // namespace timer

#endif  // ndef MCMC_TIMER_H__
