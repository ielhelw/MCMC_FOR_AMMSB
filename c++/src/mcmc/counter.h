#ifndef MCMC_COUNTER_H__
#define MCMC_COUNTER_H__

#include <iostream>
#include <string>

namespace mcmc {

class Counter {
 public:
  Counter() : count_(0), hits_(0) {
  }

  Counter(const std::string& name) : count_(0), hits_(0), name_(name) {
  }

  void tick(::size_t n = 1) {
    count_ += n;
    hits_++;
  }

  ::size_t count() const {
    return count_;
  }

  ::size_t hits() const {
    return hits_;
  }

  std::ostream &put(std::ostream &s) const {
    s << std::left << std::setw(36) << name_ << " " << count_ << " / " <<
      hits_ << " = " << std::right;
    if (hits_ > 0) {
      s << ((1.0 * count_) / hits_);
    } else {
      s << "n/a";
    }
    return s;
  }

 private:
  ::size_t count_;
  ::size_t hits_;
  std::string name_;
};

std::ostream &operator<<(std::ostream &s, const Counter &counter) {
  return counter.put(s);
}

}

#endif
