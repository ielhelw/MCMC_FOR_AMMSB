#ifndef MCMC_RANDOM_H__
#define MCMC_RANDOM_H__

#include <cassert>
#include <cstdlib>

#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <list>
#include <random>
#include <sstream>
#include <iostream>
#include <iomanip>

#include "mcmc/exception.h"

namespace mcmc {
namespace Random {

#ifdef RANDOM_FOLLOWS_CPP_WENZHE
#  define RANDOM_SYSTEM
#endif

// #define RANDOM_SYSTEM
// #define USE_TAUS2_RANDOM

class Random {
protected:
	Random() {
	}

public:
	Random(unsigned int seed) {
#ifndef RANDOM_SYSTEM
		if (seed == 0) throw NumberFormatException("Random seed value 0 not allowed"); // zero value not allowed
		xorshift_state[0] = seed;
		xorshift_state[1] = seed + 1;
#endif
		srand(seed);
		std::cerr << "Random seed [" << xorshift_state[0] << "," << xorshift_state[1] << "]" << std::endl;
	}

	Random(unsigned int seed_hi, unsigned int seed_lo) {
#ifndef RANDOM_SYSTEM
		if (seed_lo == 0) throw NumberFormatException("Random seed value 0 not allowed"); // zero value not allowed
		xorshift_state[0] = seed_lo;
		xorshift_state[1] = seed_hi;
#endif
		srand(seed_lo ^ seed_hi);
		std::cerr << "Random seed [" << xorshift_state[0] << "," << xorshift_state[1] << "]" << std::endl;
	}

	virtual ~Random() {
	}

#ifndef RANDOM_SYSTEM
	inline uint64_t xorshift_128plus() {
		uint64_t s1 = xorshift_state[0];
		uint64_t s0 = xorshift_state[1];
		xorshift_state[0] = s0;
		s1 ^= s1 << 23;
		return (xorshift_state[1] = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))) + s0;
	}

    uint64_t seed(int x) const {
      return xorshift_state[x];
    }

	inline uint64_t rand() {
		uint64_t r = xorshift_128plus();
		// std::cerr << "0: rand/X() " << r << std::endl;
		return r;
	}
#else

    const uint64_t seed(int x) const {
      return 0;
    }

	// use system ::rand()
#endif

	int64_t randint(int64_t from, int64_t upto) {
		return (rand() % (upto + 1 - from)) + from;
	}

	double random() {
		return (1.0 * rand() / std::numeric_limits<uint64_t>::max());
	}


	std::vector<std::vector<double> > randn(::size_t K, ::size_t N) {
		std::vector<std::vector<double> > r(K);
		for (::size_t k = 0; k < K; k++) {
			r[k] = randn(N);
		}

		return r;
	}


protected:
	std::unordered_set<int> sample(int from, int upto, ::size_t count) {
		assert((int)count <= upto - from);

		std::unordered_set<int> accu;
		for (::size_t i = 0; i < count; i++) {
			int r = randint(from, upto - 1);
			if (accu.find(r) == accu.end()) {
				accu.insert(r);
			} else {
				i--;
			}
		}

		return accu;
	}


	template <class Input, class Result, class Inserter>
	void sample(Result *result, const Input &input, ::size_t count, Inserter inserter) {
		std::unordered_set<int> accu = sample(0, (int)input.size(), count);

		::size_t c = 0;
		for (auto i: input) {
			if (accu.find(c) != accu.end()) {
				inserter(*result, i);
			}
			c++;
		}
	}


public:
	template <class List>
	List *sample(const List &population, ::size_t count) {
		List *result = new List();

		struct Inserter {
			void operator() (List &list, typename List::value_type &item) {
				list.insert(item);
			}
		};
		sample(result, population, count, Inserter());

#ifndef NDEBUG
		for (auto i : *result) {
			assert(population.find(i) != population.end());
		}
#endif

		return result;
	}


	template <class List>
	List *sample(const List *population, ::size_t count) {
		return sample(*population, count);
	}


	template <class Element>
	std::vector<Element> *sample(const std::vector<Element> &population, ::size_t count) {
		std::unordered_set<int> accu;
		std::vector<Element> *result = new std::vector<Element>(accu.size());

		struct Inserter {
			void operator() (std::vector<Element> &list, Element &item) {
				list.push_back(item);
			}
		};
		sample(result, population, count, Inserter());

		return result;
	}


	std::vector<int> *sampleRange(int N, ::size_t count) {
		auto accu = sample(0, N, count);
		return new std::vector<int>(accu.begin(), accu.end());
	}


	template <class Element>
	std::list<Element> *sampleList(const std::unordered_set<Element> &population, ::size_t count) {
		std::list<Element> *result = new std::list<Element>();
		struct Inserter {
			void operator() (std::list<Element> &list, Element &item) {
				list.push_back(item);
			}
		};
		sample(result, population, count, Inserter());

#ifndef NDEBUG
		for (auto i : *result) {
			assert(population.find(i) != population.end());
		}
#endif

		return result;
	}


	template <class Element>
	std::list<Element> *sampleList(const std::unordered_set<Element> *population, ::size_t count) {
		return sampleList(*population, count);
	}


	std::vector<std::vector<double> > gamma(double p1, double p2, ::size_t n1, ::size_t n2) {
		// std::vector<std::vector<double> > *a = new std::vector<double>(n1, std::vector<double>(n2, 0.0));
		std::vector<std::vector<double> > a(n1, std::vector<double>(n2));
#ifdef RANDOM_SYSTEM
#if __GNUC_MINOR__ >= 5
		std::gamma_distribution<double> gammaDistribution(p1, p2);

		for (::size_t i = 0; i < n1; i++) {
			for (::size_t j = 0; j < n2; j++) {
				a[i][j] = gammaDistribution(generator);
			}
		}
#else	// if __GNUC_MINOR__ >= 5
		throw UnimplementedException("random::gamma");
#endif
#else
		for (::size_t i = 0; i < n1; i++) {
			for (::size_t j = 0; j < n2; j++) {
				a[i][j] = gsl_ran_gamma (NULL, p1, p2);
			}
		}
#endif	// def RANDOM_SYSTEM
		return a;
	}

#ifndef RANDOM_SYSTEM
protected:
/* gauss.c - gaussian random numbers, using the Ziggurat method
 *
 * Copyright (C) 2005  Jochen Voss.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this library; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/*
 * This routine is based on the following article, with a couple of
 * modifications which simplify the implementation.
 *
 *     George Marsaglia, Wai Wan Tsang
 *     The Ziggurat Method for Generating Random Variables
 *     Journal of Statistical Software, vol. 5 (2000), no. 8
 *     http://www.jstatsoft.org/v05/i08/
 *
 * The modifications are:
 *
 * 1) use 128 steps instead of 256 to decrease the amount of static
 * data necessary.  
 *
 * 2) use an acceptance sampling from an exponential wedge
 * exp(-R*(x-R/2)) for the tail of the base strip to simplify the
 * implementation.  The area of exponential wedge is used in
 * calculating 'v' and the coefficients in ziggurat table, so the
 * coefficients differ slightly from those in the Marsaglia and Tsang
 * paper.
 *
 * See also Leong et al, "A Comment on the Implementation of the
 * Ziggurat Method", Journal of Statistical Software, vol 5 (2005), no 7.
 *
 */


/* position of right-most step */
#define PARAM_R 3.44428647676

/* tabulated values for the heigt of the Ziggurat levels */
static const double ytab[128];

/* tabulated values for 2^24 times x[i]/x[i+1],
 * used to accept for U*x[i+1]<=x[i] without any floating point operations */
static const uint64_t ktab[128];

/* tabulated values of 2^{-24}*x[i] */
static const double wtab[128];


#define gsl_rng_get(r)				rand()
#define gsl_rng_uniform(r)			random()
#define gsl_rng_uniform_int(r, n)	randint(0, n - 1)
struct gsl_rng;

// #define TICK(x)		do x++; while (0)
#define TICK(x)


double
gsl_ran_gaussian_ziggurat (const gsl_rng * r, const double sigma)
{
  uint64_t i, j;
  int sign;
  double x, y;

  // const unsigned long int range = r->type->max - r->type->min;
  // const unsigned long int offset = r->type->min;
  const uint64_t range = 0xffffffffUL;
  const uint64_t offset = 0;

  TICK(n_randn);

  while (1)
    {
	  TICK(iters_randn);
      if (range >= 0xFFFFFFFF)
        {
          uint64_t k = gsl_rng_get(r) - offset;
          i = (k & 0xFF);
          j = (k >> 8) & 0xFFFFFF;
        }
      else if (range >= 0x00FFFFFF)
        {
          uint64_t k1 = gsl_rng_get(r) - offset;
          uint64_t k2 = gsl_rng_get(r) - offset;
          i = (k1 & 0xFF);
          j = (k2 & 0x00FFFFFF);
        }
      else
        {
          i = gsl_rng_uniform_int (r, 256); /*  choose the step */
          j = gsl_rng_uniform_int (r, 16777216);  /* sample from 2^24 */
        }

      sign = (i & 0x80) ? +1 : -1;
      i &= 0x7f;

      x = j * this->wtab[i];

      if (j < this->ktab[i])
        break;

	  TICK(ktab_exceed_randn);
      if (i < 127)
        {
          double y0, y1, U1;
          y0 = this->ytab[i];
          y1 = this->ytab[i + 1];
          U1 = gsl_rng_uniform (r);
          y = y1 + (y0 - y1) * U1;
        }
      else
        {
          double U1, U2;
          U1 = 1.0 - gsl_rng_uniform (r);
          U2 = gsl_rng_uniform (r);
          x = PARAM_R - log (U1) / PARAM_R;
          y = exp (-PARAM_R * (x - 0.5 * PARAM_R)) * U2;
		  TICK(log_exp_randn);
        }

      if (y < exp (-0.5 * x * x))
        break;
    }

  return sign * sigma * x;
}
#endif	// def RANDOM_SYSTEM


public:
	std::vector<double> randn(::size_t K) {
		auto r = std::vector<double>(K);
#ifdef RANDOM_SYSTEM
#if __GNUC_MINOR__ >= 5
		for (::size_t i = 0; i < K; i++) {
			r[i] = normalDistribution(generator);
		}

		return r;

#else	// if __GNUC_MINOR__ >= 5
		throw UnimplementedException("random::randn");
#endif
#else
		for (::size_t i = 0; i < K; i++) {
			r[i] = gsl_ran_gaussian_ziggurat(NULL, 1.0);
		}
#endif

		return r;
	}

	void report() {
#ifndef RANDOM_SYSTEM
		if (n_randn > 0) {
			std::cerr << "randn calls " << n_randn << " iters " << (1.0 * iters_randn / n_randn) << " ktab miss " << (1.0 * ktab_exceed_randn / n_randn) << " log/exp calls " << (1.0 * log_exp_randn / n_randn) << std::endl;
		}
#endif
	}

#ifndef RANDOM_SYSTEM
protected:
	::size_t n_randn = 0;
	::size_t iters_randn = 0;
	::size_t log_exp_randn = 0;
	::size_t ktab_exceed_randn = 0;


protected:
/* rng/gsl_rng.h
 * 
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2004, 2007 James Theiler, Brian Gough
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

inline double
gsl_rng_uniform_pos (const gsl_rng * r)
{
	double x ;
	do
	{
		x = gsl_rng_uniform (r) ;
	}
	while (x == 0) ;

	return x ;
}

protected:
/* randist/gamma.c
 * 
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2007 James Theiler, Brian Gough
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/* New version based on Marsaglia and Tsang, "A Simple Method for
 * generating gamma variables", ACM Transactions on Mathematical
 * Software, Vol 26, No 3 (2000), p363-372.
 *
 * Implemented by J.D.Lamb@btinternet.com, minor modifications for GSL
 * by Brian Gough
 */

#ifdef UNUSED
double
gsl_ran_gamma_mt (const gsl_rng * r, const double a, const double b)
{
  return gsl_ran_gamma (r, a, b);
}
#endif

double
gsl_ran_gamma (const gsl_rng * r, double a, double b)
{
  /* assume a > 0 */

	double f = 1.0;
#ifdef SUPPORT_RECURSION
  if (a < 1)
    {
      double u = gsl_rng_uniform_pos (r);
      return gsl_ran_gamma (r, 1.0 + a, b) * pow (u, 1.0 / a);
    }
#else
  while (a < 1) {
      double u = gsl_rng_uniform_pos (r);
	  f = f * pow (u, 1.0 / a);
      a = 1.0 + a;
  }
#endif

  {
    double x, v, u;
    double d = a - 1.0 / 3.0;
    double c = (1.0 / 3.0) / sqrt (d);

    while (1)
      {
        do
          {
            x = gsl_ran_gaussian_ziggurat (r, 1.0);
            v = 1.0 + c * x;
          }
        while (v <= 0);

        v = v * v * v;
        u = gsl_rng_uniform_pos (r);

        if (u < 1 - 0.0331 * x * x * x * x) 
          break;

        if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
          break;
      }
    
    return f * b * d * v;
  }
}
#endif	// def RANDOM_SYSTEM


public:
	std::string state() {
#ifndef RANDOM_SYSTEM
		std::ostringstream s;
		s << "(" << xorshift_state[0] << "," << xorshift_state[1] << ")";
		return s.str();
#else
		return "<nope>";
#endif
	}


protected:
#ifndef RANDOM_SYSTEM
	uint64_t xorshift_state[2];
#else
#if __GNUC_MINOR__ >= 5
	std::default_random_engine generator;
	std::normal_distribution<double> normalDistribution;
#else  // if __GNUC_MINOR__ >= 5
	throw UnimplementedException("random::gamma");
#endif
#endif	// ndef RANDOM_SYSTEM
};


class FileReaderRandom : public Random {
public:
	FileReaderRandom(unsigned int seed) : Random(seed) {
		floatReader.open("random.random");
		intReader.open("random.randint");
		sampleReader.open("random.sample");
		choiceReader.open("random.choice");
		gammaReader.open("random.gamma");
		noiseReader.open("random.noise");
	}

	virtual ~FileReaderRandom() {
	}


protected:
	void getline(std::ifstream &f, std::string &line) {
		do {
			std::getline(f, line);
			if (! f) {
				break;
			}
		} while (line[0] == '#');

		if (! f) {
			if (f.eof()) {
				throw IOException("end of file");
			} else {
				throw IOException("file read error");
			}
		}
	}


public:
	std::vector<double> randn(::size_t K) {
		std::string line;
		std::vector<double> r(K);

		getline(noiseReader, line);
		std::istringstream is(line);
		for (::size_t k = 0; k < K; k++) {
			if (! (is >> r[k])) {
				throw IOException("end of line");
			}
		}

		std::cerr << "Read random.randn[" << K << "]" << std::endl;
		if (false) {
			for (::size_t k = 0; k < K; k++) {
				// std::cerr << r[k] << " ";
			}
			// std::cerr << std::endl;
		}

		return r;
	}


#ifdef UNNECESSARY_IF_VIRTUAL
	std::vector<std::vector<double> > randn(::size_t K, ::size_t N) {
		// std::cerr << "Read random.randn[" << K << "," << N << "]" << std::endl;
		std::vector<std::vector<double> > r(K);
		for (::size_t k = 0; k < K; k++) {
			r[k] = randn(N);
		}

		return r;
	}
#endif


	double random() {
		std::string line;
		getline(floatReader, line);

		double r;
		std::istringstream is(line);
		if (! (is >> r)) {
			throw IOException("end of line");
		}

		if (false) {
			std::cerr << "Read random.random " << r << std::endl;
		}
		return r;
	}


	int64_t randint(int64_t from, int64_t upto) {
		std::string line;
		getline(intReader, line);

		int64_t r;
		std::istringstream is(line);
		if (! (is >> r)) {
			throw IOException("end of line");
		}

		// std::cerr << "Read random.randint " << r << std::endl;
		return r;
	}


	template <class List>
	List *sample(const List &population, ::size_t count) {
		std::string line;
		List *result = new List();
		getline(sampleReader, line);

		std::istringstream is(line);

		for (::size_t i = 0; i < count; i++) {
			typename List::key_type key(is);
			result->insert(key);
		}

		// std::cerr << "Read " << count << " random.sample<List> values" << std::endl;
		return result;
	}


	template <class List>
	List *sample(const List *population, ::size_t count) {
		return sample(*population, count);
	}


	template <class Element>
	std::vector<Element> *sample(const std::vector<Element> &population, ::size_t count) {
		std::string line;
		getline(sampleReader, line);
		std::istringstream is(line);
		// // std::cerr << "Read vector<something>[" << count << "] sample; input line '" << is.str() << "'" << std::endl;

		std::vector<Element> *result = new std::vector<Element>(count);

		for (::size_t i = 0; i < count; i++) {
			int r;

			if (! (is >> r)) {
				throw IOException("end of line");
			}
			result->push_back(r);
		}
		// std::cerr << "Read " << count << " random.sample<vector> values" << std::endl;

		return result;
	}


	std::vector<int> *sampleRange(int N, ::size_t count) {
		std::vector<int> dummy;

		return sample(dummy, count);
	}


	template <class Element>
	std::list<Element> *sampleList(const std::unordered_set<Element> &population, ::size_t count) {
		std::string line;
		auto *result = new std::list<Element>();
		getline(sampleReader, line);

		std::istringstream is(line);

		for (::size_t i = 0; i < count; i++) {
			Element key(is);
			result->push_back(key);
		}

		std::cerr << "Read " << count << " random.sampleList<> values" << std::endl;
		return result;
	}


	template <class Element>
	std::list<Element> *sampleList(const std::unordered_set<Element> *population, ::size_t count) {
		return sampleList(*population, count);
	}


	std::vector<std::vector<double> > gamma(double p1, double p2, ::size_t n1, ::size_t n2) {
		std::vector<std::vector<double> > a(n1, std::vector<double>(n2));

		std::string line;

		for (::size_t i = 0; i < n1; i++) {
			getline(gammaReader, line);

			std::istringstream is(line);
			for (::size_t j = 0; j < n2; j++) {
				if (! (is >> a[i][j])) {
					throw IOException("end of line");
				}
			}
		}
		// std::cerr << "Read random.gamma[" << n1 << "x" << n2 << "] values" << std::endl;

		return a;
	}

	std::ifstream floatReader;
	std::ifstream intReader;
	std::ifstream sampleReader;
	std::ifstream choiceReader;
	std::ifstream gammaReader;
	std::ifstream noiseReader;
};


#ifdef RANDOM_FOLLOWS_PYTHON
extern FileReaderRandom *random;
#else
extern Random *random;
#endif

}	// namespace Random
}	// namespace mcmc

#endif	// ndef MCMC_RANDOM_H__
