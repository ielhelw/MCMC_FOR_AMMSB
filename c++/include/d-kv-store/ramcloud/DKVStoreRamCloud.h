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

#ifndef APPS_MCMC_D_KV_STORE_RAMCLOUD_DKV_STORE_H__
#define APPS_MCMC_D_KV_STORE_RAMCLOUD_DKV_STORE_H__

#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic push
#include <RamCloud.h>
#pragma GCC diagnostic pop

#include <d-kv-store/DKVStore.h>

namespace DKV {
namespace DKVRamCloud {

class DKVStoreRamCloud : public DKVStoreInterface {

 public:
  typedef DKVStoreInterface::KeyType KeyType;
  typedef DKVStoreInterface::ValueType ValueType;

  ~DKVStoreRamCloud();

  virtual void Init(::size_t value_size, ::size_t total_values,
                    ::size_t max_capacity,
                    const std::vector<std::string> &args);

  virtual void ReadKVRecords(std::vector<ValueType *> &cache,
							 const std::vector<KeyType> &key,
                             RW_MODE::RWMode rw_mode);

  virtual void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value);

  virtual void FlushKVRecords(const std::vector<KeyType> &key);

  virtual void PurgeKVRecords();

 private:
  RAMCloud::RamCloud *client_ = NULL;
  uint64_t table_id_;
  std::string table_ = "table1";
  std::unordered_map<KeyType, RAMCloud::Tub<RAMCloud::ObjectBuffer>*> obj_buffer_map_;
};

} // namespace DKVRamCloud
} // namespace DKV

#endif  // def APPS_MCMC_D_KV_STORE_RAMCLOUD_DKV_STORE_H__
