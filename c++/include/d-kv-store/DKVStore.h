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

#ifndef __INTEL_COMPILER
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic push     
#endif
#include <boost/program_options.hpp>
#ifndef __INTEL_COMPILER
#pragma GCC diagnostic pop
#endif


namespace DKV {

namespace TYPE {
enum TYPE {
    FILE,
#ifdef ENABLE_RAMCLOUD
    RAMCLOUD,
#endif
#ifdef ENABLE_RDMA
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
#ifdef ENABLE_RAMCLOUD
	} else if (token == "ramcloud") {
		dkv_type = DKV::TYPE::RAMCLOUD;
#endif
#ifdef ENABLE_RDMA
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
#ifdef ENABLE_RAMCLOUD
	case DKV::TYPE::RAMCLOUD:
		s << "ramcloud";
		break;
#endif
#ifdef ENABLE_RDMA
	case DKV::TYPE::RDMA:
		s << "rdma";
		break;
#endif
	}

	return s;
}

}   // namespace TYPE


namespace RW_MODE {
  enum RWMode {
    READ_ONLY,
    READ_WRITE,
  };
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

class DKVStoreInterface {

 public:
  typedef int32_t KeyType;
  typedef double ValueType;

  virtual ~DKVStoreInterface() {
  }

  virtual void Init(::size_t value_size, ::size_t total_values,
                    ::size_t max_cache_capacity, ::size_t max_write_capacity,
                    const std::vector<std::string> &args) {
    value_size_ = value_size;
    total_values_ = total_values;
    cache_buffer_.Init(value_size * max_cache_capacity);
    write_buffer_.Init(value_size * max_write_capacity);
  }

  virtual bool include_master() {
    return true;
  }

  virtual void barrier() {
    throw DKVException("Unimplemented " + std::string(__func__));
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
   * Write key/value pairs.
   * @param value
   *    may be a mix of user-allocated data and buffers obtained through
   *    GetWriteKVRecords(); in the latter case, may be a zerocopy operation.
   * @reentrant: no
   */
  virtual void WriteKVRecords(const std::vector<KeyType> &key,
                              const std::vector<const ValueType *> &value) = 0;

  /**
   * Zerocopy write interface. Obtain a vector of value pointers to fill.
   * Written out by a call to WriteKVRecords, which performs the binding from
   * key to value.
   * @reentrant: no
   */
  virtual std::vector<ValueType *> GetWriteKVRecords(::size_t n) = 0;

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

  Buffer<ValueType> cache_buffer_;
  Buffer<ValueType> write_buffer_;

  std::unordered_map<KeyType, ValueType *> value_of_;
};

} // namespace DKV

#endif  // def APPS_MCMC_D_KV_STORE_H__
