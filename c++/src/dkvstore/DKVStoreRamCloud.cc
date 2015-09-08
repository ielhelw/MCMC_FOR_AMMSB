/*
 * Copyright notice
 */

#include "dkvstore/DKVStoreRamCloud.h"

#include <cassert>
#include <sstream>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace DKV {
namespace DKVRamCloud {

DKVStoreRamCloudOptions::DKVStoreRamCloudOptions()
  : table_("0.0.0.0"), proto_("infrc"),
    host_("0.0.0.0"), port_("1100"),
    desc_("Ramcloud options") {
  desc_.add_options()
    ("ramcloud.table,t", po::value<std::string>(&table_)->default_value("0.0.0.0"), "Coordinator table")
    ("ramcloud.coordinator,c", po::value<std::string>(&host_)->default_value("0.0.0.0"), "Coordinator host")
    ("ramcloud.port,p", po::value<std::string>(&port_)->default_value("1100"), "Coordinator port")
    ("ramcloud.protocol,P", po::value<std::string>(&proto_)->default_value("infrc"), "Coordinator protocol") 
    ;
}

void DKVStoreRamCloudOptions::Parse(const std::vector<std::string> &args) {
  po::variables_map vm;
  po::basic_command_line_parser<char> clp(args);
  clp.options(desc_);
  po::store(clp.run(), vm);
  po::notify(vm);
}


DKVStoreRamCloud::DKVStoreRamCloud(const std::vector<std::string> &args)
    : DKVStoreInterface(args) {
  std::cerr << "DKVStoreRamCloud args ";
  for (auto a : args) {
    std::cerr << a << " ";
  }
  std::cerr << std::endl;

  options_.Parse(args);
}

DKVStoreRamCloud::~DKVStoreRamCloud() {
  delete client_;
}

void DKVStoreRamCloud::Init(::size_t value_size, ::size_t total_values,
                            ::size_t max_cache_capacity,
                            ::size_t max_write_capacity) {
  ::DKV::DKVStoreInterface::Init(value_size, total_values,
                                 max_cache_capacity, max_write_capacity);

  std::ostringstream coordinator;
  coordinator << options_.proto() << ":host=" << options_.host() << ",port=" << options_.port();
  std::cerr << "coordinator description: " << coordinator.str() << std::endl;

  client_ = new RAMCloud::RamCloud(coordinator.str().c_str());
  try {
    table_id_ = client_->getTableId(options_.table().c_str());
  } catch (RAMCloud::TableDoesntExistException& e) {
    table_id_ = client_->createTable(options_.table().c_str(), 1);
  }
}

void DKVStoreRamCloud::ReadKVRecords(std::vector<ValueType *> &cache,
                                     const std::vector<KeyType> &key,
                                     RW_MODE::RWMode rw_mode) {
  // vals place holder
  std::vector<RAMCloud::Tub<RAMCloud::ObjectBuffer> *> bufs(key.size());
  for (::size_t i = 0; i < key.size(); ++i) {
	// FIXME: what if key[i] is already in the cache?
    bufs[i] = new RAMCloud::Tub<RAMCloud::ObjectBuffer>();
    if (rw_mode == RW_MODE::READ_ONLY) {
      obj_buffer_map_[key[i]] = bufs[i];
    }
  }
  // batch requests
  std::vector<RAMCloud::MultiReadObject*> reqs(key.size());
  for (::size_t i = 0; i < key.size(); ++i) {
    reqs[i] = new RAMCloud::MultiReadObject(table_id_,
                                            &key[i], sizeof key[i],
                                            bufs[i]);
  }
  client_->multiRead(reqs.data(), key.size());

  for (::size_t i = 0; i < key.size(); ++i) {
    /**     
     * BufferObject contains key and value data. Need to fetch the value only
     * bufs[i]->copy(0, K*sizeof(ValueType), vals[i].data());
     */
    assert((*bufs[i])->getValue() != NULL);
    if (rw_mode == RW_MODE::READ_ONLY) {
      cache[i] = (ValueType *)(*bufs[i])->getValue();
    } else {
      ValueType *cache_pointer = cache_buffer_.get(value_size_);
      cache[i] = cache_pointer;
      value_of_[key[i]] = cache_pointer;
      memcpy(cache[i], (*bufs[i])->getValue(), value_size_ * sizeof(ValueType));
    }
  }

  for (auto e : reqs) {
    delete e;
  }
  if (rw_mode != RW_MODE::READ_ONLY) {
    for (auto b : bufs) {
      delete b;
    }
  }
}

void DKVStoreRamCloud::WriteKVRecords(const std::vector<KeyType> &key,
                                      const std::vector<const ValueType *> &value) {
  std::vector<RAMCloud::MultiWriteObject *> req(key.size());
  for (::size_t i = 0; i < key.size(); i++) {
    req[i] = new RAMCloud::MultiWriteObject(table_id_,
                                            &key[i], sizeof key[i],
                                            value[i],
                                            value_size_ * sizeof(ValueType));
  }
  client_->multiWrite(req.data(), key.size());

  for (auto e : req) {
    delete e;
  }
}

std::vector<DKVStoreRamCloud::ValueType *> DKVStoreRamCloud::GetWriteKVRecords(::size_t n) {
  std::vector<ValueType *> w(n);
  for (::size_t i = 0; i < n; i++) {
    w[i] = write_buffer_.get(value_size_);
  }

  return w;
}

void DKVStoreRamCloud::FlushKVRecords(const std::vector<KeyType> &key) {
  std::vector<const ValueType *> value(key.size());
  for (::size_t i = 0; i < key.size(); i++) {
    value[i] = value_of_[key[i]];
  }
  DKVStoreRamCloud::WriteKVRecords(key, value);
  write_buffer_.reset();
}

void DKVStoreRamCloud::PurgeKVRecords() {
  // Clear the cached read-only buffers
  for (auto b : obj_buffer_map_) {
    delete b.second;
  }
  obj_buffer_map_.clear();

  // Clear the copied read/write buffer(s)
  cache_buffer_.reset();
  write_buffer_.reset();
  value_of_.clear();
}

} // namespace DKVRamCloud
} // namespace DKV
