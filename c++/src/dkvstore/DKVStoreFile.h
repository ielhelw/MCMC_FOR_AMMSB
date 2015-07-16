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

#ifndef APPS_MCMC_D_KV_STORE_FILE_DKV_STORE_H__
#define APPS_MCMC_D_KV_STORE_FILE_DKV_STORE_H__

#include "dkvstore/DKVStore.h"

namespace DKV {
namespace DKVFile {

class DKVStoreFile : public DKVStoreInterface {

 public:
  typedef DKVStoreInterface::KeyType KeyType;
  typedef DKVStoreInterface::ValueType ValueType;

  DKVStoreFile() {
  }

  virtual ~DKVStoreFile();

  virtual void Init(::size_t value_size, ::size_t total_values,
                    ::size_t max_cache_capacity, ::size_t max_write_capacity,
                    const std::vector<std::string> &args);

  virtual void ReadKVRecords(std::vector<ValueType *> &cache,
							 const std::vector<KeyType> &key,
                             RW_MODE::RWMode rw_mode);

  virtual void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value);

  virtual std::vector<ValueType *> GetWriteKVRecords(::size_t n);

  virtual void FlushKVRecords(const std::vector<KeyType> &key);

  virtual void PurgeKVRecords();

 private:
  const std::string PiFileName(KeyType node) const;
  void CreateDirNameOf(const std::string &filename) const;
  void WriteKVRecord(const KeyType key, const ValueType *cached);

  std::string file_base_;
  std::string dir_;
};

} // namespace DKVFile
} // namespace DKV

#endif  // def APPS_MCMC_D_KV_STORE_FILE_DKV_STORE_H__
