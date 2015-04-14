/*
 * Copyright notice
 */

#include "d-kv-store/file/DKVStoreFile.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <cassert>

#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic push
#include <boost/program_options.hpp>
#pragma GCC diagnostic pop

#include <mr/fileio.h>

namespace po = boost::program_options;

namespace DKV {
namespace DKVFile {

DKVStoreFile::~DKVStoreFile() {
  if (inputFileSystem_ != NULL) {
    inputFileSystem_->close();
  }
  delete[] cache_;
}

void DKVStoreFile::Init(::size_t value_size, ::size_t total_values,
                        ::size_t max_capacity,
                        const std::vector<std::string> &args) {
  value_size_ = value_size;
  total_values_ = total_values;
  max_capacity_ = max_capacity;

  std::cerr << "DKVStoreFile::Init args ";
  for (auto a : args) {
    std::cerr << a << " ";
  }
  std::cerr << std::endl;

  po::options_description desc("D-KV File options");
  desc.add_options()
    ("dkv:file:filebase,b", po::value<std::string>(&file_base_)->default_value("pi"), "File base")
    ("dkv:file:dir,d", po::value<std::string>(&dir_)->default_value(""), "Directory")
    ;

  po::variables_map vm;
  po::basic_command_line_parser<char> clp(args);
  clp.options(desc);
  po::store(clp.run(), vm);
  po::notify(vm);

  inputFileSystem_ = mr::FileSystem::openFileSystem(mr::FILESYSTEM::REGULAR);
  cache_ = new ValueType[max_capacity_ * value_size];
}

void DKVStoreFile::ReadKVRecords(std::vector<ValueType *> &cache,
                                 const std::vector<KeyType> &key,
                                 RW_MODE::RWMode rw_mode) {
  assert(next_free_ + key.size() <= max_capacity_);
  assert(cache.size() >= key.size());
  for (::size_t i = 0; i < key.size(); i++) {
    ValueType *cache_pointer = cache_ + next_free_ * value_size_;
    next_free_++;
    mr::FileReader *reader = inputFileSystem_->createReader(PiFileName(key[i]));
    reader->readFully(cache_pointer, value_size_ * sizeof(ValueType));
    cache[i] = cache_pointer;
    value_of_[key[i]] = cache_pointer;
	assert(value_of_.size() <= max_capacity_);
    delete reader;
  }
}

void DKVStoreFile::WriteKVRecords(const std::vector<KeyType> &key,
                                  const std::vector<const ValueType *> &value) {
  assert(value.size() >= key.size());
  for (::size_t i = 0; i < key.size(); ++i) {
    WriteKVRecord(key[i], value[i]);
  }
}

void DKVStoreFile::FlushKVRecords(const std::vector<KeyType> &key) {
  for (::size_t i = 0; i < key.size(); ++i) {
    WriteKVRecord(key[i], value_of_[key[i]]);
  }
}

void DKVStoreFile::PurgeKVRecords() {
  next_free_ = 0;
  value_of_.clear();
}

void DKVStoreFile::WriteKVRecord(const KeyType key,
                                 const ValueType *cached) {
  mr::FileWriter *writer = inputFileSystem_->createWriter(PiFileName(key), O_TRUNC);
  writer->write(cached, value_size_ * sizeof(ValueType));
  delete writer;
}

const std::string DKVStoreFile::PiFileName(KeyType node) const {
  const int kMaxDigits = 4;
  std::ostringstream s;
  if (dir_ != "") {
    s << dir_ << "/";
  }
  s << file_base_ << "/";
  for (int digit = kMaxDigits - 1; digit > 0; --digit) {
    int h_digit = (node >> (4 * digit)) & 0xF;
    s << std::hex << h_digit << "/";
  }
  s << std::dec << node;

  return s.str();
}

} // namespace DKVFile
} // namespace DKV
