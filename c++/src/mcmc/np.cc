#include "mcmc/np.h"

namespace mcmc {
namespace np {

#ifdef MCMC_EFFICIENCY_COMPATIBILITY_MODE
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
