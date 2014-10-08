#ifndef __GRAPH_H__
#define __GRAPH_H__

typedef struct {
    global int* edges;
    global int2* node_edges;
} _Graph_internal;

typedef struct {
    _Graph_internal _g;
} Graph;

// get number of peers for node
inline int graph_peers_count(
        global Graph *g,
        int id) {
    return g->_g.node_edges[id].x;
}

inline int graph_peers_offset(
        global Graph *g,
        int id) {
    return g->_g.node_edges[id].y;
}

inline int graph_get_peer(
        global Graph *g,
        int id,
        int peerIdx /* value from 0 to graph_peers_count() */) {
    return g->_g.edges[graph_peers_offset(g, id) + peerIdx];
}

int graph_has_peer(global Graph *g, int u, int v) {
	int2 desc = g->_g.node_edges[u];
	global int *p = g->_g.edges + desc.y;
    for (int i = 0; i < desc.x; ++i,++p) {
    	if (*p > v) return 0;
    	else if (*p == v) return 1;
    }
    return 0;
}

#endif /* __GRAPH_H__ */
