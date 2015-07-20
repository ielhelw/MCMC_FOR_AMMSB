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
		print_mem_usage(std::cerr);

		boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf;

		if (compressed_) {
			inbuf.push(boost::iostreams::gzip_decompressor());
		}
		inbuf.push(infile);
		std::istream instream(&inbuf);

		std::string line;
		std::string header;

		// start from the 5th line.
		for (int i = 0; i < 4; i++) {
			std::getline(instream, line);
			header = header + line + "\n";
		}

		mcmc::NetworkGraph *E = new mcmc::NetworkGraph();	// store all pair of edges.
		::size_t N;

		if (contiguous_) {
			Vertex max = -1;
			std::unordered_set<Vertex> vertex;
			::size_t count = 0;
			while (std::getline(instream, line)) {
				Vertex a;
				Vertex b;
				std::istringstream iss(line);
				if (! (iss >> a >> b)) {
					throw mcmc::IOException("Fail to parse int");
				}
				assert(a != b);
				a -= contiguous_offset_;
				b -= contiguous_offset_;
				assert(a >= 0);
				assert(b >= 0);
				vertex.insert(a);
				vertex.insert(b);
				max = std::max(a, max);
				max = std::max(b, max);
				Edge e(std::min(a, b), std::max(a, b));
				e.insertMe(E);
				if (progress_ > 0) {
					count++;
				   	if (count % progress_ == 0) {
						std::cerr << "Edges read " << count << std::endl;
						print_mem_usage(std::cerr);
					}
				}
			}
			std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms read unordered set" << std::endl;
			print_mem_usage(std::cerr);
			N = max + 1;
			if (max + 1 != (Vertex)vertex.size()) {
				std::ostringstream s;
				for (::size_t i = 0; i < N; i++) {
					if (vertex.find(i) == vertex.end()) {
						std::cerr << "Missing vertex: " << i << std::endl;
					}
				}
				s << "# vertices " << vertex.size() << " max vertex " << max;
				print_mem_usage(std::cerr);
			}

		} else {
			// std::set<Vertex> vertex;	// ordered set	WHY????????????????
			std::unordered_set<Vertex> vertex;
			std::vector<mcmc::Edge> edge;
			Vertex max = std::numeric_limits<Vertex>::min();
			Vertex min = std::numeric_limits<Vertex>::max();
			::size_t count = 0;
			while (std::getline(instream, line)) {
				Vertex a;
				Vertex b;
				std::istringstream iss(line);
				if (! (iss >> a >> b)) {
					throw mcmc::IOException("Fail to parse Vertex");
				}
				vertex.insert(a);
				vertex.insert(b);
				edge.push_back(Edge(a, b));
				max = std::max(max, a);
				min = std::min(min, b);
				if (progress_ > 0) {
					count++;
				   	if (count % progress_ == 0) {
						std::cerr << "Edges read " << count << std::endl;
						print_mem_usage(std::cerr);
					}
				}
			}
			std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms read ordered set" << std::endl;
			print_mem_usage(std::cerr);
			std::cerr << "#nodes " << vertex.size() <<
				" min " << min << " max " << max <<
			   	" #edges " << edge.size() << std::endl;

			std::vector<Vertex> nodelist(vertex.begin(), vertex.end()); // use range constructor, retain order

			N = nodelist.size();

			// change the node ID to make it start from 0
			std::unordered_map<Vertex, Vertex> node_id_map;
			Vertex i = 0;
			for (auto node_id: nodelist) {
				node_id_map[node_id] = i;
				i++;
			}
			std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms create map" << std::endl;
			print_mem_usage(std::cerr);

			::size_t duplicates = 0;
			::size_t self_links = 0;
			count = 0;
			for (auto i: edge) {
				Vertex node1 = node_id_map[i.first];
				Vertex node2 = node_id_map[i.second];
				Edge eIdent(i.first, i.second);
				if (node1 == node2) {
					std::cerr << "Self-link " << eIdent << ": ignore" << std::endl;
					self_links++;
					continue;
				}
				Edge e(std::min(node1, node2), std::max(node1, node2));
				if (e.in(*E)) {
					// std::cerr << "Duplicate link " << eIdent << ": ignore" << std::endl;
					duplicates++;
				} else {
					e.insertMe(E);
				}
				if (progress_ > 0) {
					count++;
				   	if (count % progress_ == 0) {
						std::cerr << "Edges inserted " << count << std::endl;
						print_mem_usage(std::cerr);
					}
				}
			}
			std::cerr << "#edges original " << edge.size() << " undirected subsets " << E->size() << " duplicates " << duplicates << " self-links " << self_links << std::endl;
			std::cerr << duration_cast<milliseconds>((system_clock::now() - start)).count() << "ms create NetworkGraph" << std::endl;
			print_mem_usage(std::cerr);
#ifdef EDGESET_IS_ADJACENCY_LIST
			for (::size_t i = 0; i < E->size(); i++) {
				if ((*E)[i].size() == 0) {
					std::cerr << "Find no edges to/from " << i << std::endl;
				}
			}
#endif
		}

		infile.close();

		print_mem_usage(std::cerr);

		return new Data(NULL, E, N, header);
	}

private:
	Vertex contiguous_offset_ = 0;

};

}	// namespace preprocess
}	// namespace mcmc

#endif	// ndef MCMC_PREPROCESS_RELATIVITY_H__
