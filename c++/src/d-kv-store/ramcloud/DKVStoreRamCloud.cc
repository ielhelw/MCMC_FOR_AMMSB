/*
 * Copyright notice
 */

#include "d-kv-store/ramcloud/DKVStoreRamCloud.h"

#include <cassert>
#include <sstream>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace DKV {
namespace DKVRamCloud {

void DKVStoreRamCloud::Init(::size_t value_size, ::size_t total_values,
                            const std::vector<std::string> &args) {
  value_size_ = value_size;
  total_values_ = total_values;

  std::string table;
  std::string proto;
  std::string host;
  std::string port;

  std::cerr << "DKVStoreRamCloud::Init args ";
  for (auto a : args) {
    std::cerr << a << " ";
  }
  std::cerr << std::endl;

  po::options_description desc("Ramcloud options");
  desc.add_options()
    ("ramcloud.table,t", po::value<std::string>(&table)->default_value("0.0.0.0"), "Coordinator table")
    ("ramcloud.coordinator,c", po::value<std::string>(&host)->default_value("0.0.0.0"), "Coordinator host")
    ("ramcloud.port,p", po::value<std::string>(&port)->default_value("1100"), "Coordinator port")
    ("ramcloud.protocol,P", po::value<std::string>(&proto)->default_value("infrc"), "Coordinator protocol") 
    ;

  po::variables_map vm;
  po::basic_command_line_parser<char> clp(args);
  clp.options(desc);
  po::store(clp.run(), vm);
  po::notify(vm);

  std::ostringstream coordinator;
  coordinator << proto << ":host=" << host << ",port=" << port;
  std::cerr << "coordinator description: " << coordinator.str() << std::endl;

  client_ = new RAMCloud::RamCloud(coordinator.str().c_str());
  try {
    table_id_ = client_->getTableId(table.c_str());
  } catch (RAMCloud::TableDoesntExistException& e) {
    table_id_ = client_->createTable(table.c_str(), 1);
  }
}

void DKVStoreRamCloud::ReadKVRecords(
    const std::vector<KeyType> &key,
    const std::vector<ValueType *> &cache) {
  // vals place holder
  std::vector<RAMCloud::Tub<RAMCloud::ObjectBuffer>> bufs(key.size());
  // batch requests
  std::vector<RAMCloud::MultiReadObject*> reqs(key.size());
  for (::size_t i = 0; i < key.size(); ++i) {
    reqs[i] = new RAMCloud::MultiReadObject(table_id_,
                                            &key[i], sizeof key[i],
                                            &bufs[i]);
  }
  client_->multiRead(reqs.data(), key.size());

  for (::size_t i = 0; i < key.size(); ++i) {
    /**     
     * BufferObject contains key and value data. Need to fetch the value only
     * bufs[i]->copy(0, K*sizeof(ValueType), vals[i].data());
     */
    assert(bufs[i]->getValue() != NULL);
    memcpy(cache[i], bufs[i]->getValue(), value_size_ * sizeof *cache[i]);
  }

  for (auto e : reqs) delete e;
}

void DKVStoreRamCloud::WriteKVRecords(
    const std::vector<KeyType> &key,
    const std::vector<const ValueType *> &cached) {
  std::vector<RAMCloud::MultiWriteObject *> req(key.size());
  for (::size_t i = 0; i < key.size(); i++) {
    req[i] = new RAMCloud::MultiWriteObject(table_id_,
                                            &key[i], sizeof key[i],
                                            cached[i],
                                            value_size_ * sizeof cached[i][0]);
  }
  client_->multiWrite(req.data(), key.size());

  for (auto e : req) {
    delete e;
  }
}

} // namespace DKVRamCloud
} // namespace DKV
