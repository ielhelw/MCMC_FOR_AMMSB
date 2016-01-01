#ifndef MCMC_TYPES_H__
#define MCMC_TYPES_H__

#include <boost/program_options.hpp>

#include "mcmc/config.h"

namespace mcmc {
namespace strategy {

enum strategy {
  STRATIFIED_RANDOM_NODE,
  STRATIFIED_RANDOM_NODE_LINKS,
  STRATIFIED_RANDOM_NODE_NONLINKS,
  RANDOM_EDGE,
};


inline std::istream& operator>> (std::istream& in, strategy& strategy) {
  namespace po = boost::program_options;

  std::string token;
  in >> token;

  if (false) {
  } else if (token == "stratified-random-node") {
    strategy = STRATIFIED_RANDOM_NODE;
  } else if (token == "stratified-random-node-links") {
    strategy = STRATIFIED_RANDOM_NODE_LINKS;
  } else if (token == "stratified-random-node-nonlinks") {
    strategy = STRATIFIED_RANDOM_NODE_NONLINKS;
  } else if (token == "random-edge") {
    strategy = RANDOM_EDGE;
  } else {
    throw po::validation_error(po::validation_error::invalid_option_value,
                               "Unknown strategy");
  }

  return in;
}


inline std::ostream& operator<< (std::ostream& s, const strategy& strategy) {
  switch (strategy) {
    case STRATIFIED_RANDOM_NODE:
      s << "stratified-random-node";
      break;
    case STRATIFIED_RANDOM_NODE_LINKS:
      s << "stratified-random-node-links";
      break;
    case STRATIFIED_RANDOM_NODE_NONLINKS:
      s << "stratified-random-node-nonlinks";
      break;
    case RANDOM_EDGE:
      s << "random-edge";
      break;
  }

  return s;
}


}	// namespace learner
}	// namespace mcmc

#endif	// ndef MCMC_TYPES_H__
