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
#include <unordered_map>

namespace DKV {

namespace RW_MODE {
  enum RWMode {
    READ_ONLY,
    READ_WRITE,
  };
}

class DKVStoreInterface {

 public:
  typedef int32_t KeyType;
  typedef double ValueType;

  virtual ~DKVStoreInterface() {
  }

  virtual void Init(::size_t value_size, ::size_t total_values,
                    ::size_t max_capacity,
                    const std::vector<std::string> &args) {
    value_size_ = value_size;
    total_values_ = total_values;
    max_capacity_ = max_capacity;
  }

  /**
   * Populate the cache area with the values belonging to @argument keys,
   * in the same order.
   * @return pointers into our cached area.
   * @reentrant: no
   */
  virtual void ReadKVRecords(std::vector<ValueType *> &cache,
							 const std::vector<KeyType> &key,
                             RW_MODE::RWMode rw_mode) = 0;

  /**
   * Write the values that belong to new keys
   * @reentrant: no
   */
  virtual void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value) = 0;

  /**
   * Write back the values that belong to rw-cached keys
   * @reentrant: no
   */
  virtual void FlushKVRecords(const std::vector<KeyType> &key) = 0;

  /**
   * Purge the cache area
   * @reentrant: no
   */
  virtual void PurgeKVRecords() = 0;

 protected:
  ::size_t value_size_;
  ::size_t total_values_;
  ::size_t max_capacity_;

  ValueType *cache_ = NULL;
  ::size_t next_free_ = 0;
  std::unordered_map<KeyType, ValueType *> value_of_;

  const ::size_t CACHE_INCREMENT = 1024;
};

} // namespace DKV

#endif  // def APPS_MCMC_D_KV_STORE_H__
