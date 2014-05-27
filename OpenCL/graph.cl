#include "graph.h"

// Graph initializer
kernel void graph_init(
		global Graph* g,
		global int* edges,
		global int2* node_edges) {
	g->_g.edges = edges;
	g->_g.node_edges = node_edges;
}
