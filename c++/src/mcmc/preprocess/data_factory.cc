#include "mcmc/preprocess/data_factory.h"

#include <boost/filesystem.hpp>

namespace mcmc {
namespace preprocess {

DataFactory::DataFactory(const Options &options)
    : dataset_class_(options.input_class_),
      filename_(options.input_filename_),
      compressed_(options.input_compressed_),
      contiguous_(options.input_contiguous_),
      dump_nodemap_file_(options.dump_nodemap_file_) {
  if (false) {
  } else if (dataset_class_ == "rcz") {
    compressed_ = true;
    contiguous_ = true;
    dataset_class_ = "relativity";
  } else if (dataset_class_ == "rz") {
    compressed_ = true;
    dataset_class_ = "relativity";
  } else if (dataset_class_ == "rc") {
    contiguous_ = true;
    dataset_class_ = "relativity";
  } else if (dataset_class_ == "gz") {
    compressed_ = true;
    dataset_class_ = "sparsehash";
  } else if (dataset_class_ == "preprocessed") {
    compressed_ = true;
    dataset_class_ = "sparsehash";
    filename_ = filename_ + "/graph.gz";
  }
}

DataFactory::~DataFactory() {}

static std::vector<Vertex> node_list(const std::unordered_map<Vertex, Vertex> &node_map) {
  std::vector<Vertex> node_vector(node_map.size(), -1);
  for (auto x: node_map) {
    if (x.second >= static_cast<Vertex>(node_vector.size())) {
      std::cerr << "Ouch, node " << x.second << " out of range" << std::endl;
    }
    if (node_vector[x.second] != -1) {
      std::cerr << "Ouch, node_vector[" << x.second << "] occurs multiple times" << std::endl;
    }
    node_vector[x.second] = x.first;
  }
  for (auto n: node_vector) {
    if (n == -1) {
      std::cerr << "Ouch, uninitialized element" << std::endl;
    }
  }
  return node_vector;
}

void DataFactory::dump_nodemap(const std::unordered_map<Vertex, Vertex> &node_map) const {
  std::ofstream save;
  boost::filesystem::path dir(dump_nodemap_file_);
  boost::filesystem::create_directories(dir.parent_path());
  save.open(dump_nodemap_file_, std::ios::out);
  std::cerr << "Save nodemap to file " << dump_nodemap_file_ << std::endl;
  std::cerr <<
    "Format: <local id> (contiguous and sorted) <blank> <input file id>" <<
    std::endl;
  auto nl = node_list(node_map);
  Vertex i = 0;
  for (auto x: nl) {
    save << i << " " << x << std::endl;
    ++i;
  }
  save.close();
}


const mcmc::Data *DataFactory::get_data() const {
  // FIXME: who will delete dataObj?
  // FIXME: solve with.... !!! templating !!! FIXME
  DataSet *dataObj = NULL;
  if (false) {
  } else if (dataset_class_ == "netscience") {
    dataObj = new NetScience(filename_);
  } else if (dataset_class_ == "relativity") {
    dataObj = new Relativity(filename_);
  } else if (dataset_class_ == "sparsehash") {
    dataObj = new SparseHashGraph(filename_);
#if 0
  } else if (dataset_class_ == "hep_ph") {
    dataObj = new HepPH(filename_);
  } else if (dataset_class_ == "astro_ph") {
    dataObj = new AstroPH(filename_);
  } else if (dataset_class_ == "condmat") {
    dataObj = new CondMat(filename_);
  } else if (dataset_class_ == "hep_th") {
    dataObj = new HepTH(filename_);
#endif
  } else {
    throw MCMCException("Unknown dataset name \"" + dataset_class_ + "\"");
  }
  dataObj->setCompressed(compressed_);
  dataObj->setContiguous(contiguous_);
  dataObj->setProgress(progress_);

  auto data = dataObj->process();
  if (dump_nodemap_file_ != "") {
    auto nodemap = dataObj->nodeMap();
    dump_nodemap(nodemap);
  }

  return data;
}

void DataFactory::setCompressed(bool on) { compressed_ = on; }

void DataFactory::setContiguous(bool on) { contiguous_ = on; }

void DataFactory::setProgress(::size_t progress) { progress_ = progress; }

void DataFactory::deleteData(const mcmc::Data *data) { delete const_cast<Data *>(data); }

}  // namespace preprocess
}  // namespace mcmc
