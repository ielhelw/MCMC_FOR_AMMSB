#ifndef MCMC_NP_H__
#define MCMC_NP_H__

#include <cmath>

#include <algorithm>


namespace mcmc {
namespace np {

template <typename Type>
static std::vector<std::vector<Type> > row_sum(const std::vector<std::vector<Type> > &a) {
	return r(std::tranform(std::accumulate(a));
}


// diff2 = np.sum(np.abs(phi_ba - phi_ba_old))
template <typename Type>
static Type sum_abs(const std::vector<Type> &a, const std::vector<Type> &b) {
	Type diff = static_cast<Type>(0);
	for (::size_t i = 0; i < a.size(); i++) {
		diff += std::abs(a[i] - b[i]);
	}

	return diff;
}


template <typename Type>
static Type sum(const std::vector<Type> &a) {
#if 0
	Type sum = 0.0;
	for (::size_t i = 0; i < a.size(); i++) {
		sum += a[i];
	}

	return sum;
#else
	return std::accumulate(a.begin(), a.end(), static_cast<Type>(0));
#endif
}

}	// namespace np
}	// namespace mcmc

#endif	// ndef MCMC_NP_H__
