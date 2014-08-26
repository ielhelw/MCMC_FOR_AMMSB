#ifndef MCMC_NP_H__
#define MCMC_NP_H__

#include <cmath>

#include <algorithm>


namespace mcmc {
namespace np {


template <typename T>
class SelectColumn {
public:
	SelectColumn(int j) : j(j) {
	}

	T operator() (const std::vector<T> &v) {
		return v[j];
	}

protected:
	int j;
};


template <typename T>
class DivideBy {
public:
	DivideBy(T d) : d(d) {
	}

	T operator() (const T &v) {
		return v / d;
	}

protected:
	T d;
};


/**
 * r[i,j] = a[i,j] / s[i] where s[i] = sum_j a[i,j]
 */
template <typename T>
static void row_normalize(std::vector<std::vector<T> > *r,
						  const std::vector<std::vector<T> > &a) {
	for (::size_t i = 0; i < a.size(); i++) {
		T row_sum = std::accumulate(a[i].begin(), a[i].end(), (T)0);
		std::transform(a[i].begin(), a[i].end(), (*r)[i].begin(), np::DivideBy<T>(row_sum));
	}
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
