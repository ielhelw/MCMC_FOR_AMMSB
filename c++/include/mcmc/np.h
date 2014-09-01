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
 *
 * r = a / np.sum(a, 1)[:,np:newaxis]
 */
template <typename T>
static void row_normalize(std::vector<std::vector<T> > *r,
						  const std::vector<std::vector<T> > &a) {
	for (::size_t i = 0; i < a.size(); i++) {
		T row_sum = sum(a[i]);
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
	return std::accumulate(a.begin(), a.end(), static_cast<Type>(0));
}


template <typename Type>
static Type sum(const std::vector<Type> *a) {
	return sum<Type>(*a);
}

template <typename Type>
static void copy2D(std::vector<std::vector<Type>> *to, const std::vector<std::vector<Type>> &from) {
	to->resize(from.size());
	for (::size_t i = 0; i < from[i].size(); i++) {
		(*to)[i] = from[i];
	}
}

}	// namespace np
}	// namespace mcmc

#endif	// ndef MCMC_NP_H__
