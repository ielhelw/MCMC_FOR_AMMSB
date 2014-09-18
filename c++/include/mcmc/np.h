#ifndef MCMC_NP_H__
#define MCMC_NP_H__

#include <cmath>
#include <cassert>

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



template <typename Type>
Type sum(const std::vector<Type> &a) {
	return std::accumulate(a.begin(), a.end(), static_cast<Type>(0));
}


template <typename T>
void normalize(std::vector<T> &r, const std::vector<T> &a) {
	struct DivideBy {
		DivideBy(T d) : d(d) {
		}

		T operator() (const T &v) {
			return v / d;
		}

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
	// No: requires normalize to return its result. What about for_each w/ 2 arguments?
	// std::transform(a.begin(), a.end(), r->begin(), normalize<T>);
	for (::size_t i = 0; i < a.size(); i++) {
		// T row_sum = sum(a[i]);
		// std::transform(a[i].begin(), a[i].end(), (*r)[i].begin(), DivideBy(row_sum));
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

#ifdef EFFICIENCY_FOLLOWS_PYTHON
std::vector<int> xrange(int from, int upto) {
	std::vector<int> r(upto - from);
	for (int i = 0; i < upto - from; i++) {
		r[i] = from + i;
	}
	return r;
}
#endif

}	// namespace np
}	// namespace mcmc

#endif	// ndef MCMC_NP_H__
