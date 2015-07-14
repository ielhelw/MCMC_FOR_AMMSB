/*
 * Copyright notice goes here
 */

/*
 * @author Rutger Hofman, VU Amsterdam
 * @author Wenzhe Li
 *
 * @date 2014-08-6
 */

#ifndef MCMC_PREPROCESS_NETSCIENCE_H__
#define MCMC_PREPROCESS_NETSCIENCE_H__

#include <fstream>
#include <unordered_map>

#include <tinyxml2.h>

#include "mcmc/data.h"
#include "mcmc/preprocess/dataset.h"

namespace tinyxml2 {

class XMLException : public std::exception {
 public:
  XMLException(const std::string &reason) throw();

  virtual ~XMLException() throw();

  virtual const char *what() const throw();

 protected:
  std::string reason;
};

}  // namespace tinyxml2

namespace mcmc {
namespace preprocess {

using namespace tinyxml2;

typedef std::unordered_map<Vertex, std::string> VertexAttrib;

class NetScience : public DataSet {
 public:
  NetScience(const std::string &filename);

  virtual ~NetScience();

  /**
   * The netscience data is stored in xml format. The function just reads all
   * the vertices
   * and edges.
   * * if vertices are not record as the format of 0,1,2,3....., we need to do
   * some
   * process.  Fortunally, there is no such issue with netscience data set.
   */
  virtual const mcmc::Data *process();
};

}  // namespace preprocess
}  // namespace mcmc

#endif  // ndef MCMC_PREPROCESS_NETSCIENCE_H__
