#ifndef MCMC_TYPES_H__
#define MCMC_TYPES_H__

#include <boost/program_options.hpp>

#include "mcmc/config.h"

namespace mcmc {
namespace strategy {

enum strategy {
  RANDOM_PAIR_NONLINKS,
  RANDOM_PAIR_LINKS,
  RANDOM_PAIR,
  RANDOM_NODE_NONLINKS,
  RANDOM_NODE_LINKS,
  RANDOM_NODE,
};


inline std::istream& operator>> (std::istream& in, strategy& strategy) {
  namespace po = boost::program_options;

  std::string token;
  in >> token;

  if (false) {
  } else if (token == "random-pair-nonlinks") {
    strategy = RANDOM_PAIR_NONLINKS;
  } else if (token == "random-pair-links") {
    strategy = RANDOM_PAIR_LINKS;
  } else if (token == "random-pair") {
    strategy = RANDOM_PAIR;
  } else if (token == "random-node-links") {
    strategy = RANDOM_NODE_LINKS;
  } else if (token == "random-node-nonlinks") {
    strategy = RANDOM_NODE_NONLINKS;
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
    case RANDOM_PAIR_NONLINKS:
      s << "random-pair-nonlinks";
      break;
    case RANDOM_PAIR_LINKS:
      s << "random-pair-links";
      break;
    case RANDOM_PAIR:
      s << "random-pair";
      break;
    case RANDOM_NODE_LINKS:
      s << "random-node-links";
      break;
    case RANDOM_NODE_NONLINKS:
      s << "random-node-nonlinks";
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
