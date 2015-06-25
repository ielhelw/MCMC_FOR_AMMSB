/*
 * Copyright notice goes here
 */

/*
 * @author Rutger Hofman, VU Amsterdam
 * @author Wenzhe Li
 *
 * @date 2014-08-6
 */

#ifndef MCMC_PREPROCESS_RELATIVITY_H__
#define MCMC_PREPROCESS_RELATIVITY_H__

#include <unordered_set>
#include <fstream>
#include <chrono>

#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "mcmc/data.h"
#include "mcmc/preprocess/dataset.h"

namespace mcmc {
namespace preprocess {

// FIXME: identical: hep_ph relativity ...

/**
 * Process relativity data set
 */
class Relativity : public DataSet {
public:
	Relativity(const std::string &filename)
			: DataSet(filename == "" ? "datasets/CA-GrQc.txt" : filename) {
	}

	virtual ~Relativity() {
	}

	/**
	 * The data is stored in .txt file. The format of data is as follows, the first column
	 * is line number. Within each line, it is tab separated.
	 *
	 * [1] some texts
	 * [2] some texts
	 * [3] some texts
	 * [4] some texts
	 * [5] 1    100
	 * [6] 1    103
	 * [7] 4    400
	 * [8] ............
	 *
	 * However, the node ID is not increasing by 1 every time. Thus, we re-format
	 * the node ID first.
	 */
	virtual const Data *process() {
		using namespace std::chrono;
		auto start = system_clock::now();

		std::ios_base::openmode mode = std::ios_base::in;
		if (compressed_) {
			mode |= std::ios_base::binary;
		}
		std::ifstream infile(filename_, mode);
		if (! infile) {
			throw mcmc::IOException("Cannot open " + filename_);
		}

		std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms open file" << std::endl;

		boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf;

		if (compressed_) {
			inbuf.push(boost::iostreams::gzip_decompressor());
		}
		inbuf.push(infile);
		std::istream instream(&inbuf);

		std::string line;

		// start from the 5th line.
		for (int i = 0; i < 4; i++) {
			std::getline(instream, line);
		}

		mcmc::EdgeSet *E = new mcmc::EdgeSet();	// store all pair of edges.
		::size_t N;

		if (contiguous_) {
			int max = -1;
			std::unordered_set<int> vertex;
			::size_t count = 0;
			while (std::getline(instream, line)) {
				int a;
				int b;
				std::istringstream iss(line);
				if (! (iss >> a >> b)) {
					throw mcmc::IOException("Fail to parse int");
				}
				a--;
				b--;
				vertex.insert(a);
				vertex.insert(b);
				max = std::max(a, max);
				max = std::max(b, max);
				Edge e(std::min(a, b), std::max(a, b));
				e.insertMe(E);
				count++;
				if (count % progress_ == 0) {
					std::cerr << "Read " << count << " edges" << std::endl;
				}
			}
			std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms read unordered set" << std::endl;
			N = vertex.size();
			if (max + 1 != (int)N) {
				std::ostringstream s;
				s << "# vertices " << N << " max vertex " << max;
				throw OutOfRangeException(s.str());
			}

		} else {
			// std::set<int> vertex;	// ordered set	WHY????????????????
			std::unordered_set<int> vertex;
			std::vector<mcmc::Edge> edge;
			int max = std::numeric_limits<int>::min();
			int min = std::numeric_limits<int>::max();
			::size_t count = 0;
			while (std::getline(instream, line)) {
				int a;
				int b;
				std::istringstream iss(line);
				if (! (iss >> a >> b)) {
					throw mcmc::IOException("Fail to parse int");
				}
				vertex.insert(a);
				vertex.insert(b);
				edge.push_back(Edge(a, b));
				max = std::max(max, a);
				min = std::min(min, b);
				count++;
				if (count % progress_ == 0) {
					std::cerr << "Read " << count << " edges" << std::endl;
				}
			}
			std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms read ordered set" << std::endl;
			std::cerr << "#nodes " << vertex.size() <<
				" min " << min << " max " << max <<
			   	" #vertices " << edge.size() << std::endl;

			std::vector<int> nodelist(vertex.begin(), vertex.end()); // use range constructor, retain order

			N = nodelist.size();

			// change the node ID to make it start from 0
			std::unordered_map<int, int> node_id_map;
			int i = 0;
			for (auto node_id: nodelist) {
				node_id_map[node_id] = i;
				i++;
			}
			std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms create map" << std::endl;

			for (auto i: edge) {
				int node1 = node_id_map[i.first];
				int node2 = node_id_map[i.second];
				if (node1 == node2) {
					continue;
				}
				Edge e(std::min(node1, node2), std::max(node1, node2));
				e.insertMe(E);
			}
			std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms create EdgeSet" << std::endl;
		}

		infile.close();

		return new Data(NULL, E, N);
	}

};

}	// namespace preprocess
}	// namespace mcmc

#endif	// ndef MCMC_PREPROCESS_RELATIVITY_H__
