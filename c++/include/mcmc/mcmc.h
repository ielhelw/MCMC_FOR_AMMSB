#ifndef MCMC_MCMC_H__
#define MCMC_MCMC_H__

#include "mcmc/options.h"

#include "mcmc/preprocess/data_factory.h"

#include "mcmc/learning/mcmc_sampler_stochastic.h"
#ifdef ENABLE_OPENCL
#include "mcmc/learning/mcmc_clsampler_stochastic.h"
#endif
#ifdef ENABLE_DISTRIBUTED
#include "mcmc/learning/mcmc_sampler_stochastic-distr.h"
#endif
#include "mcmc/learning/mcmc_sampler_batch.h"
#include "mcmc/learning/variational_inference_stochastic.h"
#include "mcmc/learning/variational_inference_batch.h"

#endif	// ndef MCMC_MCMC_H__
