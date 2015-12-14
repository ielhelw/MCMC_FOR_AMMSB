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
#include <string>
#include <exception>
#include <iostream>     // Warning/error report

#include <boost/program_options.hpp>

#include <dkvstore/config.h>


namespace DKV {

enum TYPE {
  FILE,
#ifdef MCMC_ENABLE_RDMA
  RDMA,
#endif
};


inline std::istream& operator>> (std::istream& in, TYPE& dkv_type) {
  namespace po = boost::program_options;

  std::string token;
  in >> token;

  if (false) {
  } else if (token == "file") {
    dkv_type = DKV::TYPE::FILE;
#ifdef MCMC_ENABLE_RDMA
  } else if (token == "rdma") {
    dkv_type = DKV::TYPE::RDMA;
#endif
  } else {
    throw po::validation_error(po::validation_error::invalid_option_value,
                               "Unknown D-KV type");
  }

  return in;
}


inline std::ostream& operator<< (std::ostream& s, TYPE& dkv_type) {
  switch (dkv_type) {
    case DKV::TYPE::FILE:
      s << "file";
      break;
#ifdef MCMC_ENABLE_RDMA
    case DKV::TYPE::RDMA:
      s << "rdma";
      break;
#endif
  }

  return s;
}


class DKVException : public std::exception {
 public:
  DKVException(const std::string &reason) throw() : reason_(reason) {
  }

  virtual ~DKVException() throw() {
  }

  virtual const char *what() const throw() {
    return reason_.c_str();
  }

 protected:
  const std::string &reason_;
};


template <typename ValueType>
class Buffer {
 public:
  Buffer() : capacity_(0), next_free_(0), buffer_(NULL), managed_(false) {
  }

  ~Buffer() {
    if (managed_) {
      delete[] buffer_;
      buffer_ = (ValueType *)0x55555555;
    }
  }

  void Init(::size_t capacity) {
    delete[] buffer_;
    capacity_ = capacity;
    buffer_ = new ValueType[capacity];
  }

  void Init(ValueType *buffer, ::size_t capacity) {
    capacity_ = capacity;
    buffer_ = buffer;
    managed_ = false;
  }

  ValueType *get(::size_t n) {
    if (n + next_free_ > capacity_) {
      throw DKVException("Buffer: get() request exceeds capacity");
    }

    ValueType *v = buffer_ + next_free_;
    next_free_ += n;

    return v;
  }

  void reset() {
    next_free_ = 0;
  }

  ValueType *buffer() const {
    return buffer_;
  }

  ::size_t capacity() const {
    return capacity_;
  }

 private:
  ::size_t capacity_;
  ::size_t next_free_;
  ValueType *buffer_;
  bool managed_;
};

class DKVStoreOptions {
 public:
  virtual void Parse(const std::vector<std::string> &args) = 0;

  virtual boost::program_options::options_description* GetMutable() = 0;
};

class DKVStoreInterface {

 public:
  typedef int32_t KeyType;
  typedef Float ValueType;

  DKVStoreInterface(const std::vector<std::string> &args) { }

  virtual ~DKVStoreInterface() {
  }

  /**
   * @param cache_buffer_capacity is capacity <strong>per buffer</strong>
   */
  virtual void Init(::size_t value_size, ::size_t total_values,
                    ::size_t num_cache_buffers, ::size_t cache_buffer_capacity,
                    ::size_t max_write_capacity) {
    value_size_ = value_size;
    total_values_ = total_values;
    num_cache_buffers_ = num_cache_buffers;
    cache_buffer_.resize(num_cache_buffers);
    for (auto b : cache_buffer_) {
      b.Init(value_size * cache_buffer_capacity);
    }
    write_buffer_.Init(value_size * max_write_capacity);
  }

  virtual bool include_master() {
    return true;
  }

  virtual void barrier() {
    static bool first = true;
    if (first) {
      std::cerr << "Unimplemented " << __func__ << std::endl;
      first = false;
    }
  }

  /**
   * Populate the cache area with the values belonging to @argument keys,
   * in the same order.
   * @return pointers into our cached area.
   * @reentrant: no
   */
  virtual void ReadKVRecords(::size_t buffer, std::vector<ValueType *> &cache,
                             const std::vector<KeyType> &key) = 0;

  /**
   * Write key/value pairs.
   * @reentrant: no
   */
  virtual void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value) = 0;

  /**
   * Flush writes from the cache area
   * @reentrant: no
   */
  virtual void FlushKVRecords() = 0;
  /**
   * Purge reads from the cache area
   * @reentrant: no
   */
  virtual void PurgeKVRecords(::size_t buffer) = 0;

 protected:
  ::size_t value_size_;
  ::size_t total_values_;
  ::size_t num_cache_buffers_;

  std::vector<Buffer<ValueType> > cache_buffer_;
  Buffer<ValueType> write_buffer_;

  std::unordered_map<KeyType, ValueType *> value_of_;
};

} // namespace DKV

#endif  // def APPS_MCMC_D_KV_STORE_H__
