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

#include <mr/fileio.h>

#include <d-kv-store/DKVStore.h>

namespace DKV {
namespace DKVFile {

class DKVStoreFile : public DKVStoreInterface {

 public:
  typedef DKVStoreInterface::KeyType KeyType;
  typedef DKVStoreInterface::ValueType ValueType;

  virtual ~DKVStoreFile() {
    if (inputFileSystem_ != NULL) {
      inputFileSystem_->close();
    }
  }

  virtual void Init(::size_t value_size, ::size_t total_values,
                    const std::vector<std::string> &args);

  /*
   * Populate the cache area with the values belonging to @argument keys,
   * in the same order
   */
  virtual void ReadKVRecords(const std::vector<KeyType> &keys,
                             const std::vector<ValueType *> &cache);

  virtual void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &cached);

 private:
  const std::string PiFileName(KeyType node) const;
  void WriteKVRecord(const KeyType key, const ValueType *cached);

  mr::FileSystem *inputFileSystem_ = NULL;
  std::string file_base_;
  std::string dir_;
};

} // namespace DKVFile
} // namespace DKV

#endif  // def APPS_MCMC_D_KV_STORE_FILE_DKV_STORE_H__
