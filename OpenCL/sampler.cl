#if __OPENCL_VERSION__ <= CL_VERSION_1_1
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#include "graph.h"
#include "random.h"

#ifndef MAX_NODE_ID
#error "Need definition of MAX_NODE_ID"
#endif

#ifndef K
#error "Need definition of K"
#endif

#ifndef NEIGHBOR_SAMPLE_SIZE
#error "Need definition of NEIGHBOR_SAMPLE_SIZE"
#endif

#define NODE_ID_VALID(NID) ((NID) >= 0 && (NID) < MAX_NODE_ID)

#if ULONG_MAX != RAND_MAX
#error ULONG_MAX " != " RAND_MAX
#endif


typedef struct {
	struct {
		global Graph *G;
		global Graph *HG;
		global int  *Nodes;
		global int  *NodesNeighbors;
		global int  *NodesNeighborsHash;
		global int2 *Edges;
		global double *Pi;
		global double *Phi;
		global double *Beta;
		global double2 *Theta;
		global double *ThetaSum;
		global double *Scratch;
		global ulong2 *RandomSeed;
		global int *errCtrl;
		global char *errMsg;
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
		global double *stored_random;
#endif
	} bufs;
} Buffers;

kernel void init_buffers(
		global Buffers *bufs,
		global Graph *G,
		global Graph *HG,
		global int  *Nodes,
		global int  *NodesNeighbors,
		global int  *NodesNeighborsHash,
		global int2 *Edges,
		global double *Pi,
		global double *Phi,
		global double *Beta,
		global double2 *Theta,
		global double *ThetaSum,
		global double *Scratch,
		global ulong2 *RandomSeed,
		global int *errCtrl,
		global char *errMsg
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
		, global double *stored_random
#endif
		) {
	bufs->bufs.G = G;
	bufs->bufs.HG = HG;
	bufs->bufs.Nodes = Nodes;
	bufs->bufs.NodesNeighbors = NodesNeighbors;
	bufs->bufs.NodesNeighborsHash = NodesNeighborsHash;
	bufs->bufs.Edges = Edges;
	bufs->bufs.Pi = Pi;
	bufs->bufs.Phi = Phi;
	bufs->bufs.Beta = Beta;
	bufs->bufs.Theta = Theta;
	bufs->bufs.ThetaSum = ThetaSum;
	bufs->bufs.Scratch = Scratch;
	bufs->bufs.RandomSeed = RandomSeed;
	bufs->bufs.errCtrl = errCtrl;
	bufs->bufs.errMsg = errMsg;
#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
	bufs->bufs.stored_random = stored_random;
#endif
}

void report_first_error(global Buffers *bufs, constant char *msg) {
	if (atomic_cmpxchg(bufs->bufs.errCtrl, 0, 1) == 0) {
		// first error report
		global char* dst = bufs->bufs.errMsg;
		while (*msg != 0) {
			*dst++ = *msg++;
		}
		*dst = 0;
	}
}

// LINEAR SEARACH
int find_linear(global int *arr, int item, int up) {
	for (int i = 0; i < up; ++i) {
		if (item == arr[i]) return 1;
	}
	return 0;
}

// LTE BINARY SEARCH
int find_le_linear(global double *arr, double item, int up, int lo) {
	int i;
	for (i = lo; i < up; ++i) {
		if (item <= arr[i]) break;
	}
	if (up == i) return -1;
	return i;
}

#define LINEAR_LIMIT 30

int find_le(global double *arr, double item, int up, int lo) {
	if (item > arr[up -1]) return -1;
	int res;
	if (up - lo < LINEAR_LIMIT) {
		res = find_le_linear(arr, item, up, lo);
	} else {
		while (up - lo > 1) {
			int m = (lo + up) / 2;
			if (item < arr[m]) {
				up = m;
			} else {
				lo = m;
			}
		}
		if (item > arr[lo]) {
			res = up;
		} else {
			res = lo;
		}
	}
	return res;
}

// HASH TABLE: DOUBLE HASHING
#define HASH_OK(val) ( val >= 0)
#define HASH_EMPTY (-1)
#define HASH_FOUND (-2)
#define HASH_FAIL (-3)
inline int hash1(const int key, const int n_buckets) {
	return key % n_buckets;
}

inline int hash2(const int key, const int n_buckets) {
	// SOME_PRIME must be smaller than NEIGHBOR_SAMPLE_SIZE
#if NEIGHBOR_SAMPLE_SIZE > 3559
	const int SOME_PRIME = 3559;
#elif NEIGHBOR_SAMPLE_SIZE > 1117
	const int SOME_PRIME = 1117;
#elif NEIGHBOR_SAMPLE_SIZE > 331
	const int SOME_PRIME = 331;
#elif NEIGHBOR_SAMPLE_SIZE > 47
	const int SOME_PRIME = 47;
#else
	const int SOME_PRIME = 3;
#endif
	return (key % SOME_PRIME) | 1; // must be odd
}

inline int hash_put(const int key, global int* buckets, const int n_buckets) {
	const int h1 = hash1(key, n_buckets);
	const int h2 = hash2(key, n_buckets);

	for (int i = 0; i < n_buckets; ++i) {
		int loc = (h1 + i*h2) % n_buckets;

		if (buckets[loc] == HASH_EMPTY) {
			buckets[loc] = key;
			return loc;
		} else if (buckets[loc] == key) {
			return HASH_FOUND;
		}
	}
	return HASH_FAIL;
}


#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
kernel void
random_gamma_dummy(global Buffers *bufs,
				   double eta0,
				   double eta1,
				   int X,
				   int Y)
{
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	ulong2 randomSeed = bufs->bufs.RandomSeed[gid];
if (gid != 0) printf((__constant char *)"OOOOOOPPPPPPPPPPPSSSSSSSSS %s\n", "random_gamma_dummy");

	for (int i = gid; i < X; i += gsize) {
		for (int j = 0; j < Y; j++) {
			(void)rand_gamma(&randomSeed, eta0, eta1);
		}
	}
	bufs->bufs.RandomSeed[gid] = randomSeed;
}
#endif


kernel void
random_gamma(global Buffers *bufs,
			 global double *data,
			 double eta0,
			 double eta1,
			 int X,
			 int Y)
{
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	ulong2 randomSeed = bufs->bufs.RandomSeed[gid];
if (gid != 0) printf((__constant char *)"OOOOOOPPPPPPPPPPPSSSSSSSSS %s\n", "random_gamma");

	for (int i = gid; i < X; i += gsize) {
		for (int j = 0; j < Y; j++) {
			data[i * Y + j] = rand_gamma(&randomSeed, eta0, eta1);
		}
	}
	bufs->bufs.RandomSeed[gid] = randomSeed;
}


kernel void
row_normalize(const global double *in,
			  global double *out,
			  int X,
			  int Y)
{
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);

	for (int i = gid; i < X; i += gsize) {
		const global double *gin = in + i * Y;
		global double *gout = out + i * Y;

		double sum = 0.0;
		for (int j = 0; j < Y; j++) {
			sum += gin[j];
		}

		for (int j = 0; j < Y; j++) {
			gout[j] = gin[j] / sum;
		}
	}
}


typedef struct PERP_ACCU {
	double		likelihood;
	int			count;
	int			padding;
} perp_accu_t;	// hope padding/alignment is the same as on the host...


inline double
cal_edge_likelihood(global const double *pi_a,
					global const double *pi_b,
					const int y,
					global const double *beta,
					const double epsilon)
{
	double s = 0.0;
	int iy = y ? 1 : 0;
	int y_1 = iy - 1;
	int y2_1 = y_1 + iy;
	double sum = 0.0;
	for (int k = 0; k < K; k++) {
		double f = pi_a[k] * pi_b[k];
		sum += f;
		s += f * (beta[k] * y2_1 - y_1);
	}
	if (! y) {
		s += (1.0 - sum) * (1.0 - epsilon);
	}

	if (s < 1.0e-30) {
		s = 1.0e-30;
	}

	return log(s);
}


// FIXME TODO make this iterable over subsets of the graph
kernel void cal_perplexity(
		global const int3 hg[],
		const int H,			// |hg|
		global const double *pi,// (#total_nodes, K)
		global const double *beta,// (#K)
		const double epsilon,
		global double *linkLikelihood,	// global_size
		global double *nonLinkLikelihood,	// global_size
		global int *linkCount,	// global_size
		global int *nonLinkCount	// global_size
		)
{
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);

	double l0 = 0.0;
	double l1 = 0.0;
	int c0 = 0;
	int c1 = 0;
	for (int i = gid; i < H; i += gsize) {
		global const int3 *edge = &hg[i];
		double el = cal_edge_likelihood(pi + (*edge).x * K,
										pi + (*edge).y * K,
										(*edge).z,
										beta,
										epsilon);
		if ((*edge).z == 1) {
			l0 += el;
			c0++;
		} else {
			l1 += el;
			c1++;
		}
	}

	linkLikelihood[gid] = l0;
	linkCount[gid] = c0;
	nonLinkLikelihood[gid] = l1;
	nonLinkCount[gid] = c1;

	// and perform a scan over scratch.{likelihood,count} + a scan over scratch'.{likelihood,count}
}


inline int sample_neighbor_nodes_of(
		global Buffers *bufs,
		const int node,
		global const Graph *hg,
		global int* neighbor_nodes,
		global int* neighbor_nodes_hash,
		ulong2* randomSeed)
{
#if 0
	if (bufs->bufs.Nodes[node] == 1218) {
		int2 desc = hg->_g.node_edges[node];
		const global int *p = hg->_g.edges + desc.y;
		printf((__constant char *)"Node %d neighbors[%d] [", bufs->bufs.Nodes[node], desc.x);
		for (int i = 0; i < desc.x; ++i,++p) {
			printf((__constant char *)"%d ", *p);
		}
		printf((__constant char *)"]\n");
	}
#endif

	for (int i = 0; i < NEIGHBOR_SAMPLE_SIZE; ++i) {
		int neighborId;
		int ret;
		for (;;) {
			neighborId = randint(randomSeed, 0, MAX_NODE_ID - 1);
			const bool cond1 = neighborId != bufs->bufs.Nodes[node];
// printf((__constant char *)"node[%d] = %d query neighbor %d\n", node, bufs->bufs.Nodes[node], neighborId);
			const bool cond2 = !graph_has_peer(hg, node, neighborId);
			const bool cond = cond1 && cond2;
			// printf((__constant char *)"node %d neighbor %d peer %d randint %d seed (%lu,%lu)\n", bufs->bufs.Nodes[node], neighborId, cond2, neighborId, (*randomSeed).x, (*randomSeed).y);
			if (cond) {
				ret = hash_put(neighborId, neighbor_nodes_hash, HASH_SIZE);
				if (HASH_OK(ret)) {
					// printf((__constant char *)"neighbor_nodes@%d [%d] := %d\n", NEIGHBOR_SAMPLE_SIZE, i, ret);
					neighbor_nodes[i] = ret;
					break;
				}
				if (ret == HASH_FOUND) continue;
				if (ret == HASH_FAIL) {
					report_first_error(bufs,( constant char* ) "ERROR: FAILED TO INSERT ITEM IN HASH\n");
					return -1;
				}
			}
		}
	}
#ifdef RANDOM_FOLLOWS_CPP // for verification only
	for (int i = 0; i < NEIGHBOR_SAMPLE_SIZE - 1; ++i) {
		for (int j = 0; j < NEIGHBOR_SAMPLE_SIZE - i - 1; ++j) {
			int loc1 = neighbor_nodes[j];
			int loc2 = neighbor_nodes[j + 1];
			if (neighbor_nodes_hash[loc1] > neighbor_nodes_hash[loc2]) {
				int tmp = neighbor_nodes_hash[loc1];
				neighbor_nodes_hash[loc1] = neighbor_nodes_hash[loc2];
				neighbor_nodes_hash[loc2] = tmp;
			}
		}
	}
#endif

#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
	printf((__constant char *)"Store an extra %d randn()\n", K);
	for (int i = 0; i < K; ++i) {
		bufs->bufs.stored_random[node * K + i] = randn(randomSeed);
	}
#endif

#if 0
		printf((__constant char *)"Neighbors: ");
		for (int i = 0; i < NEIGHBOR_SAMPLE_SIZE; ++i) {
			int x = neighbor_nodes[i];
			printf((__constant char *)"%d ", neighbor_nodes_hash[x]);
		}
		printf((__constant char *)"\n");
#endif

	for (int i = 0; i < NEIGHBOR_SAMPLE_SIZE; ++i) {
		int x = neighbor_nodes[i];
		neighbor_nodes[i] = neighbor_nodes_hash[x];
		neighbor_nodes_hash[x] = HASH_EMPTY; // reset the hash bucket to empty
	}

	return 0;
}

kernel void sample_neighbor_nodes(
		global Buffers *bufs,
		const int N) // #nodes
		{
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	ulong2 randomSeed = bufs->bufs.RandomSeed[gid];
if (gid != 0) printf((__constant char *)"OOOOOOPPPPPPPPPPPSSSSSSSSS %s\n", "sample_neighbor_nodes");

	for (int i = gid; i < N; i += gsize) {
		// int node = bufs->bufs.Nodes[i];
		int node = i;
		int ret = sample_neighbor_nodes_of(
				bufs,
				node,
				bufs->bufs.HG,
				// index by i
				bufs->bufs.NodesNeighbors + i * NEIGHBOR_SAMPLE_SIZE,
				bufs->bufs.NodesNeighborsHash + i * HASH_SIZE,
				&randomSeed);
		if (ret) break;
	}
	bufs->bufs.RandomSeed[gid] = randomSeed;
	// printf((__constant char *)"seed (%lu,%lu)\n", randomSeed.x, randomSeed.y);
}


void update_phi_for_node_(global Buffers *bufs,
		const int node,
		const int neighbors,	// neighbor subset size
		const int N,			// #nodes
		global const Graph *g,
		global const int *neighbor_nodes,
		global double *pi,// #(neighbors,K)
		global double *phi,// #K
		global double *grads, // #K
		global double *probs, // #K
		double alpha, double epsilon, // FIXME: put into bufs->params
	   	double eps_t,
		ulong2 *randomSeed
		) {
	const global double *beta = bufs->bufs.Beta;
	double noise[K];

	global double *pi_a = pi + node * K;

#if 0
		printf((__constant char *)"update_phi pre phi[%d] ", bufs->bufs.Nodes[node]);
		for (int k = 0; k < K; k++) {
			printf((__constant char *)"%.12f ", phi[k]);
		}
		printf((__constant char *)"\n");
		printf((__constant char *)"node %d=%d pi %p pi_a %p\n", node, bufs->bufs.Nodes[node], pi, pi_a);
		printf((__constant char *)"pi[%d] ", bufs->bufs.Nodes[node]);
		for (int k = 0; k < K; k++) {
			printf((__constant char *)"%.12f ", pi_a[k]);
		}
		printf((__constant char *)"\n");
		for (int n = 0; n < NEIGHBOR_SAMPLE_SIZE; n++) {
			printf((__constant char *)"pi[%d] ", neighbor_nodes[n]);
			global double *pi_b = pi + (N + node * neighbors + n) * K;
			for (int k = 0; k < K; k++) {
				printf((__constant char *)"%.12f ", pi_b[k]);
			}
			printf((__constant char *)"\n");
		}
#endif

#if 0
	if (bufs->bufs.Nodes[node] == 5) {
		printf((__constant char *)"phi[%d]: ", bufs->bufs.Nodes[node]);
		for (int i = 0; i < K; i++) {
			printf((__constant char *)"%.12f ", phi[i]);
			if ((i + 1) % 10 == 0) printf((__constant char *)"\n    ");
		}
		printf((__constant char *)"\n");
	}
#endif

	double phi_i_sum = 0;
	for (int k = 0; k < K; k++) phi_i_sum += phi[k];
	for (int k = 0; k < K; k++) {
		grads[k] = 0.0;
	}
	for (int i = 0; i < NEIGHBOR_SAMPLE_SIZE; i++) {
		int neighbor = neighbor_nodes[i];
		if (neighbor == bufs->bufs.Nodes[node]) {
			continue;
		}

		int y_ab = graph_has_peer(g, node, neighbor);
		double e = (y_ab == 1) ? epsilon : 1.0 - epsilon;
		double probs_sum = 0.0;
		global double *pi_b = pi + (N + node * neighbors + i) * K;
		for (int k = 0; k < K; k++) {
			double f = (y_ab == 1) ? (beta[k] - epsilon) : (epsilon - beta[k]);
			probs[k] = pi_a[k] * (pi_b[k] * f + e);
			probs_sum += probs[k];
		}
		for (int k = 0; k < K; k++) {
			grads[k] += (probs[k] / probs_sum) / phi[k] - 1.0 / phi_i_sum;
		}
	}
	// printf((__constant char *)"Node %d Random seed: (%lu,%lu)\n", bufs->bufs.Nodes[node], (*randomSeed).x, (*randomSeed).y);
	double Nn = (1.0 * MAX_NODE_ID) / (NEIGHBOR_SAMPLE_SIZE - 1);
	for (int k = 0; k < K; ++k) {
		if (isnan(phi[k])) {
			printf((__constant char *)"%d Oops, phi[%d] NaN\n", __LINE__, k);
		}

#ifdef RANDOM_FOLLOWS_SCALABLE_GRAPH
		noise[k] = bufs->bufs.stored_random[node * K + k];
#else
		// FIXME no need to retain noise
		noise[k] = randn(randomSeed);
#endif
		double phi_star_k = fabs(phi[k] + eps_t / 2.0
				* (alpha - phi[k] + Nn * grads[k])
				+ sqrt(eps_t * phi[k]) * noise[k]);
		phi[k] = phi_star_k;
		if (isnan(phi[k])) {
			printf((__constant char *)"%d Oops, phi[%d] NaN\n", __LINE__, k);
		}
	}
	// printf((__constant char *)"Write phi %p..%p\n", phi, phi + K);
	// EMIT({bufs->bufs.Nodes[node], phi[node]})

#if 0
		printf((__constant char *)"update_phi post Nn %.12f phi[%d] ", Nn, bufs->bufs.Nodes[node]);
		for (int k = 0; k < K; k++) {
			printf((__constant char *)"%.12f ", phi[k]);
		}
		printf((__constant char *)"\n");
		printf((__constant char *)"pi[%d] ", bufs->bufs.Nodes[node]);
		for (int k = 0; k < K; k++) {
			printf((__constant char *)"%.12f ", pi[k]);
		}
		printf((__constant char *)"\n");
		printf((__constant char *)"grads ");
		for (int k = 0; k < K; k++) {
			printf((__constant char *)"%.12f ", grads[k]);
		}
		printf((__constant char *)"\n");
		printf((__constant char *)"noise ");
		for (int k = 0; k < K; k++) {
			printf((__constant char *)"%.12f ", noise[k]);
		}
		printf((__constant char *)"\n");
#endif
}


kernel void update_phi(
		global Buffers *bufs,
		const int neighbors,
		const int N, // #nodes
		double alpha, double epsilon,	// these all in bufs->params
		double eps_t
		){
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	global double *grads = bufs->bufs.Scratch + 2 * gid * K;
	global double *prods = bufs->bufs.Scratch + (2 * N + gid) * K;
	ulong2 randomSeed = bufs->bufs.RandomSeed[gid];
if (gid != 0) printf((__constant char *)"OOOOOOPPPPPPPPPPPSSSSSSSSS %s\n", "update_phi");
	// printf((__constant char *)"seed (%lu,%lu)\n", randomSeed.x, randomSeed.y);
	for (int i = gid; i < N; i += gsize) {
		// int node = bufs->bufs.Nodes[i];
		int node = i;
		update_phi_for_node_(
				bufs,
				node,
				neighbors,
				N,
				bufs->bufs.G,
				bufs->bufs.NodesNeighbors + i * neighbors,
				bufs->bufs.Pi,
				bufs->bufs.Phi + node * K,
				grads,
			   	prods,
				alpha, epsilon,
				eps_t,
				&randomSeed
				);
	}
	bufs->bufs.RandomSeed[gid] = randomSeed;
	// printf((__constant char *)"seed (%lu,%lu)\n", randomSeed.x, randomSeed.y);
}


void update_pi_for_node(
		global Buffers *bufs,
		int node,
		global double *pi,
		global double *phi)
{
	global double *phi_a = phi + K * node;
	global double *pi_a = pi + K * node;

	double phi_i_sum = 0;
	for (int k = 0; k < K; k++) phi_i_sum += phi_a[k];
	for (int k = 0; k < K; k++) {
		pi_a[k] = phi_a[k] / phi_i_sum;
	}
#if 0
	printf((__constant char *)"phi[%d] ", node);
	for (int k = 0; k < K; k++) {
		printf((__constant char *)"%.12f ", phi_a[k]);
	}
	printf((__constant char *)"\n");
	printf((__constant char *)"pi[%d] ", node);
	for (int k = 0; k < K; k++) {
		printf((__constant char *)"%.12f ", pi_a[k]);
	}
	printf((__constant char *)"\n");
#endif
}


kernel void update_pi(
		global Buffers *bufs,
		const int N)
{
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
if (gid != 0) printf((__constant char *)"OOOOOOPPPPPPPPPPPSSSSSSSSS %s\n", "update_pi");

	for (int i = gid; i < N; i += gsize) {
		int node = i;
		update_pi_for_node(bufs, node, bufs->bufs.Pi, bufs->bufs.Phi);
	}
}


kernel void update_beta_calculate_theta_sum(
		global double2 *theta,		// #K
		global double *theta_sum	// #K
		)
{
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);

	for (int k = gid; k < K; k += gsize) {
		theta_sum[k] = theta[k].x + theta[k].y;
		// if (isnan(theta_sum[k])) {
			// printf((__constant char *)"Oopps, theta_sum[%d] isNaN\n", k);
		// }
	}
}


kernel void update_beta_calculate_grads(
		global Buffers *bufs,
		const int E,	// #edges
		const double scale,
		const int count_partial_sums,
		const double epsilon
		) {
	size_t gid = get_global_id(0);
if (gid != 0) printf((__constant char *)"OOOOOOPPPPPPPPPPPSSSSSSSSS %s\n", "update_beta_calculate_grads");

	if (gid < count_partial_sums) {
		size_t gsize = get_global_size(0);
		global double *beta = bufs->bufs.Beta;
		global double2 *grads = (global double2 *)bufs->bufs.Scratch + gid * K;
		global double *probs = (global double *)(grads + count_partial_sums * K) + gid * K;

		for (int k = 0; k < K; ++k) {
			grads[k].x = 0;
			grads[k].y = 0;
		}
		for (int e = gid; e < E; e += gsize) {
			int i = bufs->bufs.Edges[e].x;
			int j = bufs->bufs.Edges[e].y;
			int y_ab = graph_has_peer(bufs->bufs.G, i, bufs->bufs.Nodes[j]);

			double pi_sum = 0.0;
			double prob_sum = 0.0;
			global double *pi_i = bufs->bufs.Pi + i * K;
			global double *pi_j = bufs->bufs.Pi + j * K;
			for (int k = 0; k < K; k++) {
				double f = pi_i[k] * pi_j[k];
				pi_sum += f;
				if (y_ab) {
					probs[k] = beta[k] * f;
				} else {
					probs[k] = (1.0 - beta[k]) * f;
				}
				prob_sum += probs[k];
			}

			double prob_0 = (y_ab ? epsilon : (1.0 - epsilon)) * (1.0 - pi_sum);
			prob_sum += prob_0;
			for (int k = 0; k < K; k++) {
				double f = probs[k] / prob_sum;
				grads[k].x += f * ((1-y_ab) / bufs->bufs.Theta[k].x - 1.0 / bufs->bufs.ThetaSum[k]);
				grads[k].y += f * (y_ab / bufs->bufs.Theta[k].y - 1.0 / bufs->bufs.ThetaSum[k]);
			}
		}
	}
}

kernel void update_beta_calculate_theta(
		global Buffers *bufs,
		double scale,
		double eps_t,
		double2 eta,
		int count_partial_sums
		) {
	size_t gid = get_global_id(0);
	ulong2 randomSeed = bufs->bufs.RandomSeed[gid];
	global double2 *ggrads = (global double2 *)bufs->bufs.Scratch;
if (gid != 0) printf((__constant char *)"OOOOOOPPPPPPPPPPPSSSSSSSSS %s\n", "update_beta_calculate_theta");

	if (count_partial_sums == 0) {
		for (int k = 0; k < K; ++k) {
			ggrads[k].x = 0.0;
			ggrads[k].y = 0.0;
		}
	} else {
		for (int i = 1; i < count_partial_sums; ++i) {
			global double2 *grads = (global double2 *)bufs->bufs.Scratch + i * K;
			for (int k = 0; k < K; ++k) {
				ggrads[k].x += grads[k].x;
				ggrads[k].y += grads[k].y;
			}
		}
	}
	for (int k = 0; k < K; ++k) {
		// Ugly: opencl compiler does not recognise the other double2 union fields(.s[i])
		bufs->bufs.Theta[k].x = fabs(
				bufs->bufs.Theta[k].x + eps_t / 2.0
				* (eta.x - bufs->bufs.Theta[k].x
						+ scale * ggrads[k].x)
				+ sqrt(eps_t * bufs->bufs.Theta[k].x) * randn(&randomSeed));
		bufs->bufs.Theta[k].y = fabs(
				bufs->bufs.Theta[k].y + eps_t / 2.0
				* (eta.y - bufs->bufs.Theta[k].y
						+ scale * ggrads[k].y)
				+ sqrt(eps_t * bufs->bufs.Theta[k].y) * randn(&randomSeed));
		if (isnan(bufs->bufs.Theta[k].x)) {
			printf((__constant char *)"Oopps, theta[%d].x isNaN\n", k);
		}
		if (isnan(bufs->bufs.Theta[k].y)) {
			printf((__constant char *)"Oopps, theta[%d].y isNaN\n", k);
		}
	}
	bufs->bufs.RandomSeed[gid] = randomSeed;
}

// Expand the python fully, knowing this is two dimensions so we don't need
// to synchronize to pre-calculate the sum:
// 		temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
// 		self._beta = temp[:,1]
kernel void update_beta_calculate_beta(
		global Buffers *bufs
		)
{
	global double *beta = bufs->bufs.Beta;
	global double2 *theta = bufs->bufs.Theta;
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);

	for (int k = gid; k < K; k += gsize) {
		beta[k] = theta[k].y / (theta[k].x + theta[k].y);
	}
}
