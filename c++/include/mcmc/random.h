#ifndef MCMC_RANDOM_H__
#define MCMC_RANDOM_H__

#include <cstdlib>

#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>
#include <random>
#include <sstream>
#include <iostream>

#include "mcmc/exception.h"

namespace mcmc {
namespace Random {

class Random {
public:
	Random(unsigned int seed) {
		srand(seed);
	}

	virtual ~Random() {
	}

	int randint(int from, int upto) {
#if 0
		throw UnimplementedException("Random.randint");
		return -1;
#else
		return (rand() % (upto - from)) + from;
#endif
	}


	double randn() {
		throw UnimplementedException("Random.randn");
		return 0.0;
	}


protected:
	void sample(std::unordered_set<int> *accu, int from, int upto, ::size_t count) {
		for (::size_t i = 0; i < count; i++) {
			int r = randint(from, upto);
			if (accu->find(r) == accu->end()) {
				accu->insert(r);
			} else {
				i--;
			}
		}
	}


public:
	template <class List>
	List *sample(const List &list, ::size_t count) {
		List *result = new List();

#if 1
		std::unordered_set<int> accu;
		sample(&accu, 0, list.size(), count);

		::size_t c = 0;
		for (typename List::const_iterator i = list.begin(); i != list.end(); i++) {
			if (accu.find(c) != accu.end()) {
				result->insert(*i);
			}
			c++;
		}

#else
		throw UnimplementedException("Random.sample");
#endif

		return result;
	}


	template <class List>
	List *sample(const List *list, ::size_t count) {
		return sample(*list, count);
	}


	template <class Element>
	std::vector<Element> *sample(const std::vector<Element> &list, ::size_t count) {
#if 1
		std::unordered_set<int> accu;
		sample(&accu, 0, list.size(), count);

		std::vector<Element> *result = new std::vector<Element>(accu.size());

		for (std::unordered_set<int>::const_iterator i = accu.begin(); i != accu.end(); i++) {
			result->push_back(list[*i]);
		}

#else
		throw UnimplementedException("Random.sample");
#endif

		return result;
	}


	std::vector<std::vector<double> > gamma(double p1, double p2, ::size_t n1, ::size_t n2) {
#if 1
		// std::vector<std::vector<double> > *a = new std::vector<double>(n1, std::vector<double>(n2, 0.0));
		std::vector<std::vector<double> > a(n1, std::vector<double>(n2));

		std::default_random_engine generator;
		std::gamma_distribution<double> distribution(p1, p2);

		for (::size_t i = 0; i < n1; i++) {
			for (::size_t j = 0; j < n2; j++) {
				a[i][j] = distribution(generator);
			}
		}

		return a;

#else
		throw UnimplementedException("Random.gamma");

		return NULL;
#endif
	}
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
	double randn() {
		std::string line;

		getline(noiseReader, line);

		double r;
		std::istringstream is(line);
		is >> r;

		std::cerr << "Read random.randn " << r << std::endl;
		return r;
	}


	int randint(int from, int upto) {
		std::string line;
		getline(intReader, line);

		int r;
		std::istringstream is(line);
		is >> r;

		std::cerr << "Read random.int " << r << std::endl;
		return r;
	}


protected:
	void sample(std::unordered_set<int> *accu, int from, int upto, ::size_t count) {
		std::string line;
		for (::size_t i = 0; i < count; i++) {
			int r;
			getline(sampleReader, line);

			std::istringstream is(line);
		   	is >> r;
			accu->insert(r);
		}
		std::cerr << "Read " << count << " random.sample values" << std::endl;
	}


public:
	template <class List>
	List *sample(const List &list, ::size_t count) {
		std::string line;
		List *result = new List();

		for (::size_t i = 0; i < count; i++) {
			getline(sampleReader, line);

			std::istringstream is(line);
			typename List::key_type key(is);
			result->insert(key);
		}

		std::cerr << "Read " << count << " random.sample values" << std::endl;
		return result;
	}


	template <class List>
	List *sample(const List *list, ::size_t count) {
		return sample(*list, count);
	}


	template <class Element>
	std::vector<Element> *sample(const std::vector<Element> &list, ::size_t count) {
		std::unordered_set<int> accu;
		sample(&accu, 0, list.size(), count);

		std::vector<Element> *result = new std::vector<Element>(accu.size());

		for (std::unordered_set<int>::const_iterator i = accu.begin(); i != accu.end(); i++) {
			result->push_back(list[*i]);
		}

		return result;
	}


	std::vector<std::vector<double> > gamma(double p1, double p2, ::size_t n1, ::size_t n2) {
		std::vector<std::vector<double> > a(n1, std::vector<double>(n2));

		std::string line;

		for (::size_t i = 0; i < n1; i++) {
			getline(gammaReader, line);

			std::istringstream is(line);
			for (::size_t j = 0; j < n2; j++) {
				is >> a[i][j];
			}
		}
		std::cerr << "Read random.gamma[" << n1 << "x" << n2 << "] values" << std::endl;

		return a;
	}

	std::ifstream floatReader;
	std::ifstream intReader;
	std::ifstream sampleReader;
	std::ifstream choiceReader;
	std::ifstream gammaReader;
	std::ifstream noiseReader;
};


extern FileReaderRandom *random;

}	// namespace Random
}	// namespace mcmc

#endif	// ndef MCMC_RANDOM_H__
