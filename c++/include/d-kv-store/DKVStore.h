/*
 * Copyright notice
 *
 * @author Rutger Hofman rutger@cs.vu.nl
 * @date Feb 2015
 */

/*
 * Distributed Key-Value Store that offers just enough functionality to
 * support the MCMC Stochastical applications.
 *
 * This is an Interface definition.
 */

#ifndef APPS_MCMC_D_KV_STORE_H__
#define APPS_MCMC_D_KV_STORE_H__

#include <stdint.h>

#include <vector>

namespace DKV {

class DKVStoreInterface {

 public:
  typedef int32_t KeyType;
  typedef double ValueType;

  virtual ~DKVStoreInterface() {
  }

  virtual void Init(::size_t value_size, ::size_t total_values,
                    const std::vector<std::string> &args) {
    value_size_ = value_size;
    total_values_ = total_values;
  }

  /*
   * Populate the cache area with the values belonging to @argument keys,
   * in the same order
   */
  virtual void ReadKVRecords(const std::vector<KeyType> &keys,
                             const std::vector<ValueType *> &cache) = 0;

  virtual void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &cached) = 0;

 protected:
  ::size_t value_size_;
  ::size_t total_values_;
};

} // namespace DKV

#endif  // def APPS_MCMC_D_KV_STORE_H__
