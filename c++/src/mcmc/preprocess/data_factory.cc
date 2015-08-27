#include "mcmc/preprocess/data_factory.h"

namespace mcmc {
namespace preprocess {

DataFactory::DataFactory(const Options &options)
    : dataset_class_(options.input_class_),
      filename_(options.input_filename_),
      compressed_(options.input_compressed_),
      contiguous_(options.input_contiguous_) {
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

  return dataObj->process();
}

void DataFactory::setCompressed(bool on) { compressed_ = on; }

void DataFactory::setContiguous(bool on) { contiguous_ = on; }

void DataFactory::setProgress(::size_t progress) { progress_ = progress; }

void DataFactory::deleteData(const mcmc::Data *data) { delete const_cast<Data *>(data); }

}  // namespace preprocess
}  // namespace mcmc
