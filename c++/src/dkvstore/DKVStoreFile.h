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

class DKVStoreFileOptions : public DKVStoreOptions {
 public:
  DKVStoreFileOptions();

  void Parse(const std::vector<std::string> &args) override;
  
  boost::program_options::options_description* GetMutable() override {
    return &desc_;
  }
  
  inline const std::string& file_base() const { return file_base_; }
  inline const std::string& dir() const { return dir_; }

 private:
  std::string file_base_;
  std::string dir_;
  boost::program_options::options_description desc_;

  friend std::ostream& operator<<(std::ostream& out,
                                  const DKVStoreFileOptions& opts);
};

inline std::ostream& operator<<(std::ostream& out,
                                const DKVStoreFileOptions& opts) {
  out << opts.desc_;
  return out;
}

class DKVStoreFile : public DKVStoreInterface {

 public:
  typedef DKVStoreInterface::KeyType KeyType;
  typedef DKVStoreInterface::ValueType ValueType;

  DKVStoreFile(const std::vector<std::string> &args);

  virtual ~DKVStoreFile();

  virtual void Init(::size_t value_size, ::size_t total_values,
                    ::size_t num_cache_buffers, ::size_t cache_buffer_capacity,
                    ::size_t max_write_capacity);

  virtual void ReadKVRecords(::size_t buffer, std::vector<ValueType *> &cache,
                             const std::vector<KeyType> &key);

  virtual void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value);

  virtual void PurgeKVRecords();
  virtual void PurgeKVRecords(::size_t buffer);

 private:
  const std::string PiFileName(KeyType node) const;
  void CreateDirNameOf(const std::string &filename) const;
  void WriteKVRecord(const KeyType key, const ValueType *cached);

  DKVStoreFileOptions options_;
};

} // namespace DKVFile
} // namespace DKV

#endif  // def APPS_MCMC_D_KV_STORE_FILE_DKV_STORE_H__
