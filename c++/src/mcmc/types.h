#ifndef MCMC_TYPES_H__
#define MCMC_TYPES_H__

namespace mcmc {
namespace strategy {

enum strategy {
	RANDOM_PAIR,
	RANDOM_NODE,
	STRATIFIED_RANDOM_PAIR,
	STRATIFIED_RANDOM_NODE,
};

}	// namespace learner
}	// namespace mcmc

#endif	// ndef MCMC_TYPES_H__
