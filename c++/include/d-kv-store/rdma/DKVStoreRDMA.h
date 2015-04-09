/*
 * Copyright
 */
#ifndef APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__
#define APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__

#include <vector>
#include <string>

#include <d-kv-store/DKVStore.h>

namespace DKV {
namespace DKVRDMA {

/*
 * Class description
 */
class DKVStoreRDMA : public DKVStoreInterface {

 public:
  typedef DKVStoreInterface::KeyType   KeyType;
  typedef DKVStoreInterface::ValueType ValueType;

  virtual ~DKVStoreRDMA() {
  }

  virtual void Init(::size_t value_size, ::size_t total_values,
                    const std::vector<std::string> &args);

  virtual void ReadKVRecords(const std::vector<KeyType> &keys,
                             const std::vector<ValueType *> &cache);

  virtual void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &cached);

 private:
  int32_t HostOf(KeyType key);

  ValueType *cache_;
  ValueType *store_;

  std::vector<std::string> hostname_;
  std::vector<uint16_t> port_;
};

}   // namespace DKVRDMA
}   // namespace DKV

#endif  // ndef APPS_MCMC_D_KV_STORE_RDMA_RDMA_DKVSTORE_H__
