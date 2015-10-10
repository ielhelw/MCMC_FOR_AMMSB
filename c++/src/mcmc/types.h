#ifndef MCMC_TYPES_H__
#define MCMC_TYPES_H__

#include <boost/program_options.hpp>

#include "mcmc/config.h"

namespace mcmc {
namespace strategy {

enum strategy {
  RANDOM_PAIR_NONLINKED,
  RANDOM_PAIR_LINKED,
  RANDOM_PAIR,
  RANDOM_NODE_NONLINKED,
  RANDOM_NODE_LINKED,
  RANDOM_NODE,
};


inline std::istream& operator>> (std::istream& in, strategy& strategy) {
  namespace po = boost::program_options;

  std::string token;
  in >> token;

  if (false) {
  } else if (token == "random-pair-nonlinked") {
    strategy = RANDOM_PAIR_NONLINKED;
  } else if (token == "random-pair-linked") {
    strategy = RANDOM_PAIR_LINKED;
  } else if (token == "random-pair") {
    strategy = RANDOM_PAIR;
  } else if (token == "random-node-linked") {
    strategy = RANDOM_NODE_LINKED;
  } else if (token == "random-node-nonlinked") {
    strategy = RANDOM_NODE_NONLINKED;
  } else if (token == "random-node") {
    strategy = RANDOM_NODE;
  } else {
    throw po::validation_error(po::validation_error::invalid_option_value,
                               "Unknown strategy");
  }

  return in;
}


inline std::ostream& operator<< (std::ostream& s, strategy& strategy) {
  namespace po = boost::program_options;

  switch (strategy) {
    case RANDOM_PAIR_NONLINKED:
      s << "random-pair-nonlinked";
      break;
    case RANDOM_PAIR_LINKED:
      s << "random-pair-linked";
      break;
    case RANDOM_PAIR:
      s << "random-pair";
      break;
    case RANDOM_NODE_LINKED:
      s << "random-node-linked";
      break;
    case RANDOM_NODE_NONLINKED:
      s << "random-node-nonlinked";
      break;
    case RANDOM_NODE:
      s << "random-node";
      break;
    default:
      throw po::validation_error(po::validation_error::invalid_option_value,
                                 "Unknown strategy");
  }

  return s;
}


}	// namespace learner
}	// namespace mcmc

#endif	// ndef MCMC_TYPES_H__
