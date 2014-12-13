#ifndef MCMC_OPTIONS_H__
#define MCMC_OPTIONS_H__

#include <iostream>

#include "mcmc/exception.h"

#include <boost/program_options.hpp>


namespace mcmc {

/**
 * utility function to parse a size_t.
 * @throw NumberFormatException if arg is malformed or out of range
 */

inline ::size_t parse_size_t(const std::string &argString) {
	::size_t	n = 0;
	const char *arg = argString.c_str();
	int			base = 10;

	if (strncmp(arg, "0x", strlen("0x")) == 0 ||
	   		strncmp(arg, "0X", strlen("0X") == 0)) {
		base = 16;
		arg += 2;
	} else if (arg[0] == '0') {
		base = 8;
		arg++;
	}

	while (*arg != '\0') {
		if (base == 16 && isxdigit(*arg)) {
			int a;
			if (*arg >= '0' && *arg <= '9') {
				a = *arg - '0';
			} else if (*arg >= 'a' && *arg <= 'f') {
				a = *arg - 'a' + 10;
			} else {
				assert(*arg >= 'A' && *arg <= 'F');
				a = *arg - 'A' + 10;
			}
			if ((std::numeric_limits< ::size_t>::max() - a) / base < n) {
				throw mcmc::NumberFormatException("Overflow in parse_size_t");
			}
			n = a + n * base;
		} else if (base <= 10 && isdigit(*arg)) {
			int a = *arg - '0';
			if ((std::numeric_limits< ::size_t>::max() - a) / base < n) {
				throw mcmc::NumberFormatException("Overflow in parse_size_t");
			}
			n = a + n * base;
		} else if (strcasecmp(arg, "g") == 0 || strcasecmp(arg, "gb") == 0) {
			if ((std::numeric_limits< ::size_t>::max() >> 30) < n) {
				throw mcmc::NumberFormatException("Overflow in parse_size_t");
			}
			n *= 1ULL << 30;
			break;
		} else if (strcasecmp(arg, "m") == 0 || strcasecmp(arg, "mb") == 0) {
			if ((std::numeric_limits< ::size_t>::max() >> 20) < n) {
				throw mcmc::NumberFormatException("Overflow in parse_size_t");
			}
			n *= 1ULL << 20;
			break;
		} else if (strcasecmp(arg, "k") == 0 || strcasecmp(arg, "kb") == 0) {
			if ((std::numeric_limits< ::size_t>::max() >> 10) < n) {
				throw mcmc::NumberFormatException("Overflow in parse_size_t");
			}
			n *= 1ULL << 10;
			break;
		} else {
			std::ostringstream s;
			s << "Unknown characters in number: '" << *arg << "'";
			throw mcmc::NumberFormatException(s.str());
		}
		arg++;
	}

	return n;
}


/**
 * utility function to parse an integral.
 * @throw NumberFormatException if arg is malformed or out of range
 */

template <class T>
inline T parse(const std::string &argString) {
	const char *arg = argString.c_str();

	if (arg[0] == '-') {
		::ssize_t result = -parse_size_t(arg + 1);
		if (result < (::ssize_t)std::numeric_limits<T>::min()) {
			throw mcmc::NumberFormatException("number too small for type");
		}
		return (T)result;
	} else {
		::size_t result = parse_size_t(arg);
		if (result > (::size_t)std::numeric_limits<T>::max()) {
			throw mcmc::NumberFormatException("number too large for type");
		}
		return (T)result;
	}
}


template <>
inline float parse<float>(const std::string &argString) {
	float f;

	if (sscanf(argString.c_str(), "%f", &f) != 1) {
		throw mcmc::NumberFormatException("string is not a float");
	}

	return f;
}


template <class T>
class KMG {
public:
	KMG() {
	}

	KMG(const std::string &s) {
		f = mcmc::parse<T>(s);
	}

	T operator() () {
		return f;
	}

protected:
	T f;
};


template <class T>
inline std::istream &operator>>(std::istream &in, KMG<T> &n) {
	std::string line;
	std::getline(in, line);
	n = KMG<T>(line);
	return in;
}

}	// namespace mcmc


#include <boost/program_options.hpp>

// Specializations for boost validate (parse) calls for the most common
// integer types, so we can use shorthand like 32m for 32*(2<<20)
//
namespace boost { namespace program_options {

#define VALIDATE_SPECIALIZE(T) \
template <> \
inline void validate< T, char>(boost::any& v, const std::vector<std::basic_string<char>>& xs, \
			  T *, long) { \
	validators::check_first_occurrence(v); \
	std::basic_string<char> s(validators::get_single_string(xs)); \
	try { \
		T x = mcmc::parse< T>(s); \
		v = any(x); \
	} \
	catch (mcmc::NumberFormatException e) { \
		(void)e; \
		boost::throw_exception(invalid_option_value(s)); \
	} \
}

// VALIDATE_SPECIALIZE(::size_t)
// VALIDATE_SPECIALIZE(::ssize_t)
VALIDATE_SPECIALIZE(int32_t)
VALIDATE_SPECIALIZE(uint32_t)
VALIDATE_SPECIALIZE(int64_t)
VALIDATE_SPECIALIZE(uint64_t)

}}



namespace mcmc {

namespace po = ::boost::program_options;

class Options {
public:
	Options(int argc, char *argv[]) {
		po::options_description desc("Options");

		desc.add_options()
			("help,?", "help")

			("mcmc-st,s", "MCMC Stochastical C++")
			("mcmc-b,t", "MCMC Batch C++")
#ifdef ENABLE_OPENCL
			("mcmc-st-cl,S", "MCMC Stochastical OpenCL")
			("mcmc-b-cl,T", "MCMC Batch OpenCL")
#endif

			("alpha", po::value<double>(&alpha)->default_value(0.01), "alpha")
			("eta0", po::value<double>(&eta0)->default_value(1.0), "eta0")
			("eta1", po::value<double>(&eta1)->default_value(1.0), "eta1")

			("K,K", po::value< ::size_t>(&K)->default_value(300), "K")
			("mini-batch-size,m", po::value< ::size_t>(&mini_batch_size)->default_value(0), "mini_batch_size")
			("num-node-sample,n", po::value< ::size_t>(&num_node_sample)->default_value(0), "neighbor sample size")
			("strategy", po::value<std::string>(&strategy)->default_value("unspecified"), "sampling strategy")

			("epsilon,e", po::value<double>(&epsilon)->default_value(0.0000001), "epsilon")
			("max-iteration,x", po::value< ::size_t>(&max_iteration)->default_value(10000000), "max_iteration")
			("interval,i", po::value< ::size_t>(&interval)->default_value(0), "perplexity interval")

			("a", po::value<double>(&a)->default_value(0.01), "a")
			("b", po::value<double>(&b)->default_value(1024), "b")
			("c", po::value<double>(&c)->default_value(0.55), "c")

			("num-updates,u", po::value< ::size_t>(&num_updates)->default_value(1000), "num_updates")
			("held-out-ratio,h", po::value<double>(&held_out_ratio)->default_value(0.0), "held_out_ratio")
			("output-dir,o", po::value<std::string>(&output_dir)->default_value("."), "output_dir")

			("input-file,f", po::value<std::string>(&filename)->default_value(""), "input file")
			("class,c", po::value<std::string>(&dataset_class)->default_value("netscience"), "input class")
			("contiguous,C", po::value<bool>(&contiguous)->default_value(false), "contiguous input data")

#ifdef ENABLE_OPENCL
			("platform,p", po::value<std::string>(&openClPlatform), "OpenCL platform")
			("device,d", po::value<std::string>(&openClDevice), "OpenCL device")
			("thread-group-size,G", po::value< ::size_t>(&openclGroupSize)->default_value(1), "OpenCL thread group size")
			("num-thread-groups,g", po::value< ::size_t>(&openclNumGroups)->default_value(1), "num OpenCL thread groups")
			("buffer-size,b", po::value< ::size_t>(&openclBufferSize)->default_value(0), "OpenCL buffer size");
#endif
			;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);

		if (vm.count("help") > 0) {
			std::cout << desc << std::endl;
		}

		run.mcmc_stochastical = vm.count("mcmc-st") > 0;
		run.mcmc_batch = vm.count("mcmc-b") > 0;
#ifdef ENABLE_OPENCL
		run.mcmc_stochastical_cl = vm.count("mcmc-st-cl") > 0;
		run.mcmc_batch_cl = vm.count("mcmc-b-cl") > 0;
#endif
	}

public:
	double		alpha;
	double		eta0;
	double		eta1;
	::size_t	K;
	::size_t	mini_batch_size;
	::size_t	num_node_sample;
	std::string	strategy;
	double		epsilon;
	::size_t	max_iteration;
	::size_t	interval;

	// parameters for step size
	double		a;
	double		b;
	double		c;

	::size_t	num_updates;
	double		held_out_ratio;
	std::string	output_dir;

	std::string filename;
	std::string dataset_class;
	bool		contiguous;

#ifdef ENABLE_OPENCL
	std::string	openClPlatform;
	std::string	openClDevice;
	::size_t openclGroupSize;
	::size_t openclNumGroups;
	::size_t openclBufferSize;
#endif

	struct {
		bool	mcmc_stochastical;
		bool	mcmc_batch;
#ifdef ENABLE_OPENCL
		bool	mcmc_stochastical_cl;
		bool	mcmc_batch_cl;
#endif
	} run;
};

};	// namespace mcmc

#endif	// ndef MCMC_OPTIONS_H__
