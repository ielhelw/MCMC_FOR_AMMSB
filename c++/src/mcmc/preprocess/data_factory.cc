#include "mcmc/preprocess/data_factory.h"

namespace mcmc {
namespace preprocess {

DataFactory::DataFactory(const mcmc::Options &options)
    : dataset_class_(options.dataset_class),
      filename_(options.filename),
      compressed_(options.compressed),
      contiguous_(options.contiguous),
      progress_(0) {
  if (dataset_class_ == "rcz") {
    compressed_ = true;
    contiguous_ = true;
    dataset_class_ = "relativity";
  } else if (dataset_class_ == "rz") {
    compressed_ = true;
    dataset_class_ = "relativity";
  } else if (dataset_class_ == "rc") {
    contiguous_ = true;
    dataset_class_ = "relativity";
  }
}

DataFactory::~DataFactory() {}

const mcmc::Data *DataFactory::get_data() const {
  // FIXME: who will delete dataObj?
  // FIXME: solve with.... !!! templating !!! FIXME
  DataSet *dataObj = NULL;
  if (dataset_class_ == "netscience") {
    dataObj = new NetScience(filename_);
  } else if (dataset_class_ == "relativity") {
    dataObj = new Relativity(filename_);
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

void DataFactory::deleteData(const mcmc::Data *data) {
  delete const_cast<Data *>(data);
}

};  // namespace preprocess
};  // namespace mcmc
