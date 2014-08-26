#ifndef MCMC_ESTIMATE_PHI_H__
#define MCMC_ESTIMATE_PHI_H__

#include <cmath>

#include <boost/math/special_functions/digamma.hpp>

#include "mcmc/np.h"

namespace mcmc {

typedef std::pair<std::vector<double>, std::vector<double> >	DoubleVectorPair;

template <typename Type>
class DivBy {
public:
	DivBy(Type x) : x(x) {
	}

	Type operator()(Type a) {
		return a / x;
	}

protected:
	Type x;
};


static DoubleVectorPair sample_latent_vars_for_each_pair(int a, int b,
														 const std::vector<double> &gamma_a,
														 const std::vector<double> &gamma_b,
														 const std::vector<std::vector<double> > &lamda,
														 ::size_t K, double update_threshold, double epsilon,
														 ::size_t online_iterations, const EdgeSet &linked_edges) {
	using ::boost::math::digamma;

	std::vector<double> phi_ab(K, 1.0 / K);
	std::vector<double> phi_ba(K, 1.0 / K);

    double u = 0.0;
    bool y = false;
	if (linked_edges.find(Edge(a, b)) != linked_edges.end()) {
        y = true;
	}

	const double log_epsilon = std::log(epsilon);
	const double log_1_epsilon = std::log(1.0 - epsilon);
    // alternatively update phi_ab and phi_ba, until it converges
    // or reach the maximum iterations. 
    for (::size_t i = 0; i < online_iterations; i++) {
		std::vector<double> phi_ab_old(phi_ab);
        std::vector<double> phi_ba_old(phi_ba);
            
        // first, update phi_ab
        for (::size_t k = 0; k < K; k++) {
            if (y) {
                u = -phi_ba[k]* log_epsilon;
                phi_ab[k] = std::exp(digamma(gamma_a[k])+phi_ba[k]*\
                                        (digamma(lamda[k][0])-digamma(lamda[k][0]+lamda[k][1]))+u);
			} else {
                u = -phi_ba[k]* log_1_epsilon;
                phi_ab[k] = std::exp(digamma(gamma_a[k])+phi_ba[k]*\
                                         (digamma(lamda[k][1])-digamma(lamda[k][0]+lamda[k][1]))+u)    ;
			}
		}
        double sum_phi_ab = np::sum(phi_ab);
		DivBy<double> divBy_sum_phi_ab(sum_phi_ab);
        // phi_ab = phi_ab/sum_phi_ab;
		std::transform(phi_ab.begin(), phi_ab.end(), phi_ab.begin(), divBy_sum_phi_ab);
                
        // then update phi_ba
        for (::size_t k = 0; k < K; k++) {
            if (y) {
                u = -phi_ab[k]* log_epsilon;
                phi_ba[k] = std::exp(digamma(gamma_b[k])+phi_ab[k]*\
                                        (digamma(lamda[k][0])-digamma(lamda[k][0]+lamda[k][1]))+u);
			} else {
                u = -phi_ab[k]* log_1_epsilon;
                phi_ba[k] = std::exp(digamma(gamma_b[k])+phi_ab[k]*\
                                        (digamma(lamda[k][1])-digamma(lamda[k][0]+lamda[k][1]))+u)   ;
			}
		}
               
        double sum_phi_ba = np::sum(phi_ba);
		DivBy<double> divBy_sum_phi_ba(sum_phi_ba);
        // phi_ba = phi_ba/sum_phi_ba;
		std::transform(phi_ba.begin(), phi_ba.end(), phi_ba.begin(), divBy_sum_phi_ba);
            
        // calculate the absolute difference between new value and old value
        double diff1 = np::sum_abs(phi_ab, phi_ab_old);
        double diff2 = np::sum_abs(phi_ba, phi_ba_old);
        if (diff1 < update_threshold and diff2 < update_threshold) {
            break;
		}
	}
        
    return DoubleVectorPair(phi_ab, phi_ba);
}

}	// namespace mcmc

#endif	// ndef MCMC_ESTIMATE_PHY_H__
