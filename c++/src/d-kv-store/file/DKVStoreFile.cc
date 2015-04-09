/*
 * Copyright notice
 */

#include "d-kv-store/file/DKVStoreFile.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <cassert>

#include <boost/program_options.hpp>

#include <mr/fileio.h>

namespace po = boost::program_options;

namespace DKV {
namespace DKVFile {

void DKVStoreFile::Init(::size_t value_size, ::size_t total_values,
                        const std::vector<std::string> &args) {
  value_size_ = value_size;
  total_values_ = total_values;

  std::cerr << "DKVStoreFile::Init args ";
  for (auto a : args) {
    std::cerr << a << " ";
  }
  std::cerr << std::endl;

  po::options_description desc("D-KV File options");
  desc.add_options()
    ("filebase,b", po::value<std::string>(&file_base_)->default_value(""), "File base")
    ("dir,d", po::value<std::string>(&dir_)->default_value("pi"), "Directory")
    ;

  po::variables_map vm;
  po::basic_command_line_parser<char> clp(args);
  clp.options(desc);
  po::store(clp.run(), vm);
  po::notify(vm);

  inputFileSystem_ = mr::FileSystem::openFileSystem(mr::FILESYSTEM::REGULAR);
}

void DKVStoreFile::ReadKVRecords(
    const std::vector<KeyType> &key,
    const std::vector<ValueType *> &cache) {
  for (::size_t i = 0; i < key.size(); i++) {
    mr::FileReader *reader = inputFileSystem_->createReader(PiFileName(key[i]));
    reader->read(cache[i], value_size_ * sizeof(ValueType));
    delete reader;
  }
}

void DKVStoreFile::WriteKVRecords(
    const std::vector<KeyType> &key,
    const std::vector<const ValueType *> &cached) {
  for (::size_t i = 0; i < key.size(); ++i) {
    WriteKVRecord(key[i], cached[i]);
  }
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
  s << file_base_;
  for (int digit = kMaxDigits - 1; digit > 0; --digit) {
    int h_digit = (node >> (4 * digit)) & 0xF;
    s << std::hex << h_digit << "/";
  }
  s << std::dec << node;

  return s.str();
}

} // namespace DKVFile
} // namespace DKV
