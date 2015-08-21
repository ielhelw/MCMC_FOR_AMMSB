#include "mcmc/np.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#else
int omp_get_max_threads() { return 1; }
int omp_get_thread_num() { return 0; }
int omp_get_num_threads() { return 1; }
#endif

namespace mcmc {
namespace np {

#ifdef EFFICIENCY_FOLLOWS_PYTHON
std::vector<int> xrange(int from, int upto) {
  std::vector<int> r(upto - from);
  for (int i = 0; i < upto - from; i++) {
    r[i] = from + i;
  }
  return r;
}
#endif

}  // namespace np
}  // namespace mcmc
