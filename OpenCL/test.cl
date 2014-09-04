#include "graph.h"

kernel void test_print_peers_of (global Graph *g, int n) {
	printf((__constant char*)"%d, %d\n", graph_peers_count(g, n), graph_peers_offset(g, n));
	for (int i = 0; i < graph_peers_count(g, n); ++i) {
		printf((__constant char*)"cl peer: %d\n", graph_get_peer(g, n, i));
	}
}

kernel void test_print_nodes(global int *nodes, int size) {
	for (int i = 0; i < size; ++i) {
		printf((__constant char*)" %d", nodes[i]);
	}
	printf((__constant char*)"\\n");
}

inline int getSampleNeighborAt(global int* neighbor_nodes, int u, int i) {
	return neighbor_nodes[u * NEIGHBOR_SAMPLE_SIZE + i];
}
