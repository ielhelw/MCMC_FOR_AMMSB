/*
 * Copyright notice goes here
 */

/*
 * @author Rutger Hofman, VU Amsterdam
 * @author Wenzhe Li
 *
 * @date 2014-08-6
 */

#ifndef MCMC_PREPROCESS_DATASET_H__
#define MCMC_PREPROCESS_DATASET_H__

#include "data.h"

namespace mcmc:preprocess {

/**
 * Served as the abstract base class for different types of data sets. 
 * For each data set, we should inherit from this class.
 */
template
class DataSet<class Vertex> {
public:
	/**
	 * Function to process the document. The document can be in any format. (i.e txt, xml,..)
	 * The subclass will implement this function to handle specific format of
	 * document. Finally, return the Data object can be consumed by any learner.
	 */
	/**
	 * @return the caller must delete() the result
	 */
	virtual const *mcmc::Data<Vertex> process() = NULL;
};

#endif	// ndef MCMC_PREPROCESS_DATASET_H__
