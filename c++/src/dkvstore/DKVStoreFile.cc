/*
 * Copyright notice
 */

#include "dkvstore/DKVStoreFile.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <cassert>

#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

namespace DKV {
namespace DKVFile {

DKVStoreFile::~DKVStoreFile() {
}

void DKVStoreFile::Init(::size_t value_size, ::size_t total_values,
                        ::size_t max_cache_capacity,
                        ::size_t max_write_capacity,
                        const std::vector<std::string> &args) {
	namespace po = boost::program_options;

  ::DKV::DKVStoreInterface::Init(value_size, total_values,
                                 max_cache_capacity, max_write_capacity,
                                 args);

  std::cerr << "DKVStoreFile::Init args ";
  for (auto a : args) {
    std::cerr << a << " ";
  }
  std::cerr << std::endl;

  po::options_description desc("D-KV File options");
  desc.add_options()
    ("dkv.file.filebase,b", po::value<std::string>(&file_base_)->default_value("pi"), "File base")
    ("dkv.file.dir,d", po::value<std::string>(&dir_)->default_value(""), "Directory")
    ;
  po::variables_map vm;
  po::basic_command_line_parser<char> clp(args);
  clp.options(desc);
  po::store(clp.run(), vm);
  po::notify(vm);
}

void DKVStoreFile::ReadKVRecords(std::vector<ValueType *> &cache,
                                 const std::vector<KeyType> &key,
                                 RW_MODE::RWMode rw_mode) {
  assert(cache.size() >= key.size());
  for (::size_t i = 0; i < key.size(); i++) {
    ValueType *cache_pointer = cache_buffer_.get(value_size_);

    std::string pi_file = PiFileName(key[i]);
    std::ifstream reader(pi_file.c_str());
    reader.read(reinterpret_cast<char *>(cache_pointer),
                value_size_ * sizeof(ValueType));
    cache[i] = cache_pointer;
    value_of_[key[i]] = cache_pointer;
  }
}

void DKVStoreFile::WriteKVRecords(const std::vector<KeyType> &key,
                                  const std::vector<const ValueType *> &value) {
  assert(value.size() >= key.size());
  for (::size_t i = 0; i < key.size(); ++i) {
    WriteKVRecord(key[i], value[i]);
  }
}

std::vector<DKVStoreFile::ValueType *> DKVStoreFile::GetWriteKVRecords(::size_t n) {
  std::vector<ValueType *> w(n);
  for (::size_t i = 0; i < n; i++) {
    w[i] = write_buffer_.get(value_size_);
  }

  return w;
}

void DKVStoreFile::FlushKVRecords(const std::vector<KeyType> &key) {
  for (::size_t i = 0; i < key.size(); ++i) {
    WriteKVRecord(key[i], value_of_[key[i]]);
  }
}

void DKVStoreFile::PurgeKVRecords() {
  cache_buffer_.reset();
  write_buffer_.reset();
  value_of_.clear();
}

void DKVStoreFile::WriteKVRecord(const KeyType key,
                                 const ValueType *cached) {
  std::string pi_file = PiFileName(key);
  CreateDirNameOf(pi_file);
  std::ofstream writer(pi_file.c_str(),
                       std::ios_base::out | std::ios_base::trunc);
  writer.write(reinterpret_cast<const char *>(cached),
               value_size_ * sizeof(ValueType));
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

void DKVStoreFile::CreateDirNameOf(const std::string &filename) const {
	boost::filesystem::path path(filename);
	boost::filesystem::path dirname = path.parent_path();
	if (! boost::filesystem::exists(dirname)) {
		boost::filesystem::create_directories(dirname);
	}
}

} // namespace DKVFile
} // namespace DKV
