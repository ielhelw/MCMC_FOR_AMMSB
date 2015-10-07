#include <iostream>
#include <iomanip>
#include <set>

#include <mcmc/random.h>

struct Edge {
	Edge(int first, int second) : first(first), second(second) {
	}

	bool operator== (const Edge &peer) const {
		return first == peer.first && second == peer.second;
	}

	bool operator< (const Edge &peer) const {
		return first < peer.first || (first == peer.first && second < peer.second);
	}

	int		first;
	int		second;
};

int main(int argc, char *argv[]) {
	std::cout << std::fixed << std::setprecision(12);

	auto rgen = mcmc::Random::Random(42);

	std::cout << "gamma(1.0, 1.0):" << std::endl;
	auto ag = rgen.gamma(1.0, 1.0, 2, 4);
	for (auto & r : ag) {
		for (auto & c : r) {
			std::cout << c << " ";
		}
		std::cout << std::endl;
	}

	std::cout << "randn:" << std::endl;
	auto an = rgen.randn(2, 4);
	for (auto & r : an) {
		for (auto & c : r) {
			std::cout << c << " ";
		}
		std::cout << std::endl;
	}
	
	std::cout << "randint(3LL**30):" << std::endl;
	for (int i = 0; i < 64; ++i) {
		std::cout << rgen.randint(0, 3LL << 30) << " ";
		if ((i + 1) % 8 == 0) {
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	std::cout << "Create random graph:" << std::endl;
	std::set<Edge> graph;
	for (int i = 0; i < 64; i++) {
		auto a = rgen.randint(0, 1024);
	   	auto b = rgen.randint(0, 1024);
		graph.insert(Edge(a, b));
	}
	for (auto & e : graph) {
		std::cout << "(" << e.first << "," << e.second << ") ";
	}
	std::cout << std::endl;

	std::cout << "Sample subgraph:" << std::endl;
	auto subgraph = rgen.sample(graph, 32);
	for (auto & e : *subgraph) {
		std::cout << "(" << e.first << "," << e.second << ") ";
	}
	std::cout << std::endl;
	delete subgraph;

	return 0;
}
