#ifndef MCMC_PREPROCESS_DATA_FACTORY_H__
#define MCMC_PREPROCESS_DATA_FACTORY_H__

#include "mcmc/data.h"
#include "mcmc/preprocess/dataset.h"
#include "mcmc/preprocess/netscience.h"
#include "mcmc/preprocess/hep_ph.h"
#include "mcmc/preprocess/relativity.h"

namespace mcmc {
namespace preprocess {

class DataFactory {
public:
	DataFactory(const std::string &dataset_name, const std::string &filename = "",
			   	bool compressed = false, bool contiguous = false)
   			: dataset_name(dataset_name), filename(filename),
			  compressed(compressed), contiguous(contiguous) {
	}

	virtual ~DataFactory() {
	}

	const mcmc::Data *get_data() const {
		// FIXME: who will delete dataObj?
		// FIXME: solve with.... !!! templating !!! FIXME
		DataSet *dataObj = NULL;
		if (false) {
		} else if (dataset_name == "netscience") {
			dataObj = new NetScience(filename, compressed, contiguous);
		} else if (dataset_name == "relativity") {
			dataObj = new Relativity(filename, compressed, contiguous);
		} else if (dataset_name == "rc") {
			dataObj = new Relativity(filename, true);
		} else if (dataset_name == "hep_ph") {
			// dataObj = new HepPH(filename, compressed, contiguous);
		} else if (dataset_name == "astro_ph") {
			// dataObj = new AstroPH(filename, compressed, contiguous);
		} else if (dataset_name == "condmat") {
			// dataObj = new CondMat(filename, compressed, contiguous);
		} else if (dataset_name == "hep_th") {
			// dataObj = new HepTH(filename, compressed, contiguous);
		} else {
			throw MCMCException("Unknown dataset name \"" + dataset_name + "\"");
		}

		return dataObj->process();
	}

protected:
	std::string dataset_name;
	std::string filename;
	bool		compressed;
	bool		contiguous;
};

};	// namespace preprocess
};	// namespace mcmc

#endif	// ndef MCMC_PREPROCESS_DATA_FACTORY_H__
