#ifndef MCMC_TIMER_H__
#define MCMC_TIMER_H__

#include <iostream>
#include <chrono>

#include "mcmc/config.h"

namespace mcmc {
namespace timer {

class Timer {
 public:
  Timer(const std::string &name = "");

  inline void start() { t_start = std::chrono::high_resolution_clock::now(); }

  inline void stop() {
    t_total += std::chrono::high_resolution_clock::now() - t_start;
    N++;
  }

  inline std::chrono::high_resolution_clock::duration total() const {
    return t_total;
  }

  inline ::size_t ticks() const { return N; }

  void reset();

  static void setTabular(bool on);

  static void printHeader(std::ostream &s);

  std::ostream &put(std::ostream &s) const;

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

std::ostream &operator<<(std::ostream &s, const Timer &timer);

}  // namespace mcmc
}  // namespace timer

#endif  // ndef MCMC_TIMER_H__
