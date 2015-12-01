#ifndef MCMC_NP_H__
#define MCMC_NP_H__

#include <cmath>
#include <cassert>

#include <algorithm>
#include <limits>

#include "mcmc/config.h"

#ifdef MCMC_ENABLE_OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline int omp_get_num_threads() { return 1; }
#endif

namespace mcmc {
namespace np {

template <typename T>
class SelectColumn {
 public:
  SelectColumn(int j) : j(j) {}

  T operator()(const std::vector<T> &v) { return v[j]; }

 protected:
  int j;
};

template <typename Type>
Type sum(const std::vector<Type> &a) {
  return std::accumulate(a.begin(), a.end(), static_cast<Type>(0));
}

template <typename T>
void normalize(std::vector<T> &r, const std::vector<T> &a) {
  struct DivideBy {
    DivideBy(T d) : d(d) {}

    T operator()(const T &v) { return v / d; }

    T d;
  };

  T s = np::sum(a);
  std::transform(a.begin(), a.end(), r.begin(), DivideBy(s));
}

template <typename T>
void normalize(std::vector<T> *r, const std::vector<T> &a) {
  normalize(*r, a);
}

/**
 * r[i,j] = a[i,j] / s[i] where s[i] = sum_j a[i,j]
 *
 * r = a / np.sum(a, 1)[:,np:newaxis]
 */
template <typename T>
void row_normalize(std::vector<std::vector<T> > *r,
                   const std::vector<std::vector<T> > &a) {
  // No: requires normalize to return its result. What about for_each w/ 2
  // arguments?
  // std::transform(a.begin(), a.end(), r->begin(), normalize<T>);
  for (::size_t i = 0; i < a.size(); i++) {
    // T row_sum = sum(a[i]);
    // std::transform(a[i].begin(), a[i].end(), (*r)[i].begin(),
    // DivideBy(row_sum));
    normalize((*r)[i], a[i]);
  }
}

// diff2 = np.sum(np.abs(phi_ba - phi_ba_old))
template <typename Type>
Type sum_abs(const std::vector<Type> &a, const std::vector<Type> &b) {
  Type diff = static_cast<Type>(0);
  for (::size_t i = 0; i < a.size(); i++) {
    diff += std::abs(a[i] - b[i]);
  }

  return diff;
}

template <typename T>
static ::ssize_t find_le_linear(
    const std::vector<T> &p, T location,
    ::size_t up = std::numeric_limits< ::size_t>::max(), ::size_t lo = 0) {
  if (up == std::numeric_limits< ::size_t>::max()) {
    up = p.size();
  }

  ::size_t i;
  for (i = lo; i < up; i++) {
    if (location <= p[i]) {
      break;
    }
  }
  if (up == i) {
    return -1;
  }

  return i;
}

template <typename T>
static ::ssize_t find_le(const std::vector<T> &p, T location,
                         ::size_t up = std::numeric_limits< ::size_t>::max(),
                         ::size_t lo = 0) {
  static const ::size_t LINEAR_LIMIT = 30;

  if (up == std::numeric_limits< ::size_t>::max()) {
    up = p.size();
  }

  if (location > p[up - 1]) {
    return -1;
  }

  ::ssize_t res;
  if (up - lo < LINEAR_LIMIT) {
    res = find_le_linear(p, location, up, lo);
  } else {
#ifndef NDEBUG
    ::ssize_t lin = find_le_linear(p, location, up, lo);
#endif
    while (up - lo > 1) {
      ::size_t m = (lo + up) / 2;
      assert(m < p.size());
      if (location < p[m]) {
        up = m;
      } else {
        lo = m;
      }
    }
    if (location > p[lo]) {
      res = up;
    } else {
      res = lo;
    }
    while (res > 0 && p[res] == p[res - 1]) {
      res--;
    }
    assert(lin == res);
  }
  assert(location <= p[res]);
  assert(res == 0 || location > p[res - 1]);
  return res;
}

}  // namespace np
}  // namespace mcmc

#endif  // ndef MCMC_NP_H__
