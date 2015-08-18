#ifndef MCMC_PREPROCESS_DATA_FACTORY_H__
#define MCMC_PREPROCESS_DATA_FACTORY_H__

#include "mcmc/fileio.h"
#include "mcmc/data.h"
#include "mcmc/preprocess/dataset.h"
#include "mcmc/preprocess/netscience.h"
// #include "mcmc/preprocess/hep_ph.h"
#include "mcmc/preprocess/relativity.h"
#include "mcmc/preprocess/sparsehash-graph.h"

namespace mcmc {
namespace preprocess {

class DataFactory {
public:
	DataFactory(const Options &options)
   			: dataset_class_(options.dataset_class), filename_(options.filename),
   			  compressed_(options.compressed), contiguous_(options.contiguous) {
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

	virtual ~DataFactory() {
	}

	const mcmc::Data *get_data() const {
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

	void setCompressed(bool on) {
		compressed_ = on;
	}

	void setContiguous(bool on) {
		contiguous_ = on;
	}

	void setProgress(::size_t progress) {
		progress_ = progress;
	}

	void deleteData(const mcmc::Data *data) {
		delete const_cast<Data *>(data);
	}

protected:
	std::string dataset_class_;
	std::string filename_;
	bool		compressed_ = false;
	bool		contiguous_ = false;
	::size_t	progress_ = 0;
};

};	// namespace preprocess
};	// namespace mcmc

#endif	// ndef MCMC_PREPROCESS_DATA_FACTORY_H__
