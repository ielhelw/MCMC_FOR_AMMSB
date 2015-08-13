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

#include "mcmc/exception.h"

namespace mcmc {
namespace Random {

class Random {
public:
	Random(unsigned int seed) {
		std::cerr << "Random seed " << seed << std::endl;
		srand(seed);
	}

	virtual ~Random() {
	}

	int randint(int from, int upto) {
		return (rand() % (upto - from)) + from;
	}

	double random() {
		return (1.0 * rand() / RAND_MAX);
	}


	double randn() {
#if __GNUC_MINOR__ >= 5
		return normalDistribution(generator);
#else	// if __GNUC_MINOR__ >= 5
		throw UnimplementedException("random::randn");
#endif
	}

	std::vector<double> randn(::size_t K) {
		auto r = std::vector<double>(K);
		for (::size_t i = 0; i < K; i++) {
			r[i] = randn();
		}

		return r;

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
			int r = randint(from, upto);
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


	double gamma(double p1, double p2) {
		// std::vector<std::vector<double> > *a = new std::vector<double>(n1, std::vector<double>(n2, 0.0));
#if __GNUC_MINOR__ >= 5

		std::gamma_distribution<double> gammaDistribution(p1, p2);

		return gammaDistribution(generator);
#else	// if __GNUC_MINOR__ >= 5
		throw UnimplementedException("random::gamma");
#endif
	}


	std::vector<std::vector<double> > gamma(double p1, double p2, ::size_t n1, ::size_t n2) {
		// std::vector<std::vector<double> > *a = new std::vector<double>(n1, std::vector<double>(n2, 0.0));
		std::vector<std::vector<double> > a(n1, std::vector<double>(n2));
		for (::size_t i = 0; i < n1; i++) {
			for (::size_t j = 0; j < n2; j++) {
				a[i][j] = gamma(p1, p2);
			}
		}

		return a;
	}

protected:
#if __GNUC_MINOR__ >= 5
		std::default_random_engine generator;
		std::normal_distribution<double> normalDistribution;
#else	// if __GNUC_MINOR__ >= 5
		throw UnimplementedException("random::gamma");
#endif
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


	std::vector<std::vector<double> > randn(::size_t K, ::size_t N) {
		// std::cerr << "Read random.randn[" << K << "," << N << "]" << std::endl;
		std::vector<std::vector<double> > r(K);
		for (::size_t k = 0; k < K; k++) {
			r[k] = randn(N);
		}

		return r;
	}


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


	int randint(int from, int upto) {
		std::string line;
		getline(intReader, line);

		int r;
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
