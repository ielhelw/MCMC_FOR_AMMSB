#ifndef MCMC_PREPROCESS_DATA_FACTORY_H__
#define MCMC_PREPROCESS_DATA_FACTORY_H__

namespace mcmc::preprocess {

class DataFactory {
public:
	template
	static const mcmc::Data *get_data(const std::string &dataset_name) {
		DataSet *dataObj = NULL;
		if (false) {
		} else if (dataset_name == "netscience") {
			dataObj = new NetScience();
		} else if (dataset_name == "relativity") {
			// dataObj = new NetScience();
		} else if (dataset_name == "hep_ph") {
			dataObj = new HepPH();
		} else if (dataset_name == "astro_ph") {
			// dataObj = new AstroPH();
		} else if (dataset_name == "condmat") {
			// dataObj = new CondMat();
		} else if (dataset_name == "hep_th") {
			// dataObj = new HepTH();
		} else {
			throw MCMCException("Unknown dataset name \"" + dataset_name + "\"");
		}

		return dataObj->process();
	}
};

};	// namespace mcmc::preprocess
#endif	// ndef MCMC_PREPROCESS_DATA_FACTORY_H__
