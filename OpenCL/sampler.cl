#include "graph.h"

#ifndef MAX_NODE_ID
#error "Need definition of MAX_NODE_ID"
#endif

#ifndef K
#error "Need definition of K"
#endif

#ifndef NEIGHBOR_SAMPLE_SIZE
#error "Need definition of NEIGHBOR_SAMPLE_SIZE"
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NODE_ID_VALID(NID) ((NID) >= 0 && (NID) < MAX_NODE_ID)

// PSEUDORANDOM GENERATOR
inline ulong xorshift_128plus(ulong2 *s) {
	ulong s1 = (*s).x;
	ulong s0 = (*s).y;
	(*s).x = s0;
	s1 ^= s1 << 23;
	return ((*s).y = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))) + s0;
}

#if ULONG_MAX != RAND_MAX
#error ULONG_MAX " != " RAND_MAX
#endif

inline double random(ulong2 *s) {
	return (1.0 * xorshift_128plus(s) / ULONG_MAX);
}

inline int randint(ulong2 *s, int from, int upto) {
	return (xorshift_128plus(s) % (upto - from)) + from;
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
#define HASH_OK (0)
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
	return SOME_PRIME - (key % SOME_PRIME);
}

inline int hash_put(const int key, global int* buckets, const int n_buckets) {
	const int h1 = hash1(key, n_buckets);
	const int h2 = hash2(key, n_buckets);
	int loc = (h1) % n_buckets;

	for (int i = 0; i < n_buckets; ++i) {
		if (buckets[loc] == HASH_EMPTY) {
			buckets[loc] = key;
			return HASH_OK;
		} else if (buckets[loc] == key) {
			return HASH_FOUND;
		}
		loc = (loc + h2) % n_buckets;
	}
	return HASH_FAIL;
}


#define sample_z_ab_from_edge_expr_orig(i) \
(\
	pow(beta[i], y) * pow(1-beta[i], 1-y) * pi_a[i] * pi_b[i] \
		+ pow(epsilon, y) * pow(1-epsilon, 1-y) * pi_a[i] * (1-pi_b[i]) \
)

#define sample_z_ab_from_edge_expr_optimized(i) \
(	y == 1? \
		pi_a[i] * (pi_b[i] * (beta[i] - epsilon) + epsilon) \
	: \
		pi_a[i] * (pi_b[i] * (epsilon - beta[i]) + (1-epsilon)) \
)

#define sample_z_ab_from_edge_expr sample_z_ab_from_edge_expr_optimized

inline int sample_z_ab_from_edge(
		global const double* pi_a,
		global const double *pi_b,
		global const double *beta,
		const double epsilon, const int y,
		const double random,
		global double *p
		) {
	p[0] = sample_z_ab_from_edge_expr(0);
	for (int i = 1; i < K; ++i) {
		p[i] = p[i-1] + sample_z_ab_from_edge_expr(i);
	}

	double location = random * p[K-1];
	return find_le(p, location, K, 0);
}

inline void sample_latent_vars_of(
		const int node,
		global const Graph *g,
		global const Graph *hg,
		global int* neighbor_nodes,
		global const double *pi,
		global const double *beta,
		const double epsilon,
		global int *z, /* K elements */
		ulong2* randomSeed,
		global double *p) {
	for (int i = 0; i < K; ++i) z[i] = 0;

	for (int i = 0; i < NEIGHBOR_SAMPLE_SIZE; ++i) {
		int neighborId;
		for (;;) {
			neighborId = randint(randomSeed, 0, MAX_NODE_ID);
			if (neighborId != node
					&& !graph_has_peer(hg, node, neighborId)) {
				int ret = hash_put(neighborId, neighbor_nodes, HASH_MULTIPLE*NEIGHBOR_SAMPLE_SIZE);
				if (ret == HASH_OK) break;
				if (ret == HASH_FOUND) continue;
				if (ret == HASH_FAIL) {
					printf("ERROR: FAILED TO INSERT ITEM IN HASH\n");
					break;
				}
			}
		}
	}
#ifdef ORDER_NEIGHBORS // for verification only
	for (int i = 0; i < HASH_MULTIPLE*NEIGHBOR_SAMPLE_SIZE - 1; ++i) {
		for (int j = 0; j < HASH_MULTIPLE*NEIGHBOR_SAMPLE_SIZE - i - 1; ++j) {
			if (neighbor_nodes[j] > neighbor_nodes[j + 1]) {
				int tmp = neighbor_nodes[j];
				neighbor_nodes[j] = neighbor_nodes[j+1];
				neighbor_nodes[j+1] = tmp;
			}
		}
	}
#endif

	int found = 0;
	for (int i = 0; found < NEIGHBOR_SAMPLE_SIZE && i < HASH_MULTIPLE*NEIGHBOR_SAMPLE_SIZE; ++i) {
		int neighbor = neighbor_nodes[i];

		if (neighbor == HASH_EMPTY) {
			continue;
		}
		++found;
		neighbor_nodes[i] = HASH_EMPTY;

		int y_ab = graph_has_peer(g, node, neighbor);
		int z_ab = sample_z_ab_from_edge(
				pi + node * K, pi + neighbor * K,
				beta, epsilon, y_ab,
				random(randomSeed),
				p);
		z[z_ab] += 1;
	}
}

void update_pi_for_node_(
		int node,
		global const double *pi,// #K
		global double *piUpdate,// #K
		global double *phi,// #K
		global int *z, // #K
		global double *noise, // #K
		global double *grad, // #K
		double alpha,
		double a, double b, double c,
		int step_count, int total_node_count
		) {
	double eps_t = a * pow((1 + step_count/b), -c);
	double phi_i_sum = 0;
	for (int i = 0; i < K; ++i) phi_i_sum += phi[i];
	for (int k = 0; k < K; ++k) {
		grad[k] = -NEIGHBOR_SAMPLE_SIZE * 1/phi_i_sum;
		grad[k] += 1/phi[k] * z[k];
	}
	for (int k = 0; k < K; ++k) {
		double phi_star_k = fabs(phi[k] + eps_t/2
				* (alpha - phi[k] + (total_node_count/NEIGHBOR_SAMPLE_SIZE) * grad[k])
				+ sqrt(eps_t * phi[k]) * noise[k]);
		phi[k] = phi_star_k;
	}
	double phi_sum = 0;
	for (int i = 0; i < K; ++i) phi_sum += phi[i];
	for (int i = 0; i < K; ++i) {
		piUpdate[i] = phi[i]/phi_sum;
	}
}

kernel void sample_latent_vars_and_update_pi(
		global const Graph *g,
		global const Graph *hg,
		global const int *nodes,
		const int N, // #nodes
		global int *neighbor_nodes,// (#total_nodes, NEIGHBOR_SAMPLE_SIZE)
		global const double *pi,// (#total_nodes, K)
		global double *piUpdate,// (#total_nodes, K)
		global double *phi,// (#total_nodes, K)
		global const double *beta,// (#K)
		const double epsilon,
		global int *Z, /* (#total_nodes, K) */
		global const double *noise,
		global double *scratch, // (#nodes, K)
		double alpha,
		double a, double b, double c,
		int step_count, int total_node_count,
		global ulong2 *gRandomSeed
		){
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	global double *_p = scratch + gid * K;
	ulong2 randomSeed = gRandomSeed[gid];

	for (int i = gid; i < N; i += gsize) {
		int node = nodes[i];
		sample_latent_vars_of(
				node,
				g, hg,
				neighbor_nodes + node * HASH_MULTIPLE * NEIGHBOR_SAMPLE_SIZE,
				pi,
				beta,
				epsilon,
				Z + node * K,
				&randomSeed,
				_p);
		update_pi_for_node_(node,
				pi + node * K,
				piUpdate + node * K,
				phi + node * K,
				Z + node * K,
				noise + i * K,
				scratch + i * K,
				alpha, a, b, c, step_count, total_node_count);
	}
	gRandomSeed[gid] = randomSeed;
}

#define sample_latent_vars2_orig(k) \
( \
	pow(beta[k], y) * pow(1-beta[k], 1-y) * pi_a[k] * pi_b[k] \
)

#define sample_latent_vars2_optimized(k) \
( \
	y == 1? \
		beta[k] * pi_a[k] * pi_b[k] \
	: \
		(1-beta[k]) * pi_a[k] * pi_b[k] \
)

#define sample_latent_vars2_expr sample_latent_vars2_optimized

int sample_latent_vars2_(
		int2 edge,
		global Graph *g,
		global double *pi,// (#total_nodes, K)
		global double *beta,// (#K)
		global double *p,// #K+1
		double r
		) {
	int y = graph_has_peer(g, edge.x, edge.y);
	global double *pi_a = pi + edge.x * K;
	global double *pi_b = pi + edge.y * K;

	p[0] = sample_latent_vars2_expr(0);
	double p_sum = p[0];
	for (int k = 1; k < K; ++k) {
		const double p_k = sample_latent_vars2_expr(k);
		p[k] = p_k + p[k-1];
		p_sum += p_k;
	}
	p[K] = 1 - p_sum;
	double location = r * p[K-1];
	return find_le(p, location, K, 0);
}

kernel void sample_latent_vars2(
		global Graph *g,
		global int2 *edges,
		int E, // #edges
		global double *pi,// (#total_nodes, K)
		global double *beta,// (#K)
		global int *Z,// #edges
		global double *p,// (#edges, K+1)
		global ulong2 *gRandomSeed
		) {
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	ulong2 randomSeed = gRandomSeed[gid];

	for (int i = gid; i < E; i += gsize) {
		Z[i] = sample_latent_vars2_(edges[i],
				g,
				pi,
				beta,
				p + i * (K+1),
				random(&randomSeed));
	}
	gRandomSeed[gid] = randomSeed;
}

kernel void update_beta_calculate_grads(
		global Graph *g,
		global const int2 *edges,
		const int E, // #edges
		global const int *Z,// #edges
		global const double2 *theta,// #K
		global const double *theta_sum,// (#K)
		global double2 *ggrads,// #K,
		double scale
		) {
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	global double2 *grads = ggrads + gid * K;

	for (int i = 0; i < K; ++i) {
		grads[i].x = 0;
		grads[i].y = 0;
	}
	for (int i = gid; i < E; i += gsize) {
		int y_ab = graph_has_peer(g, edges[i].x, edges[i].y);
		int k = Z[i];
		if (k != -1) {
			grads[k].x += (1-y_ab) / theta[k].x - 1 / theta_sum[k];
			grads[k].y += y_ab / theta[k].y - 1 / theta_sum[k];
		}
	}
}

kernel void update_beta_calculate_theta(
		global double2 *theta,// #K
		global const double2 *noise,// #K
		global double2 *ggrads,// #K,
		double scale,
		double eps_t,
		double2 eta,
		int count_partial_sums
		) {
	for (int i = 1; i < count_partial_sums; ++i) {
		global double2 *grads = ggrads + i * K;
		for (int k = 0; k < K; ++k) {
			ggrads[k].x += grads[k].x;
			ggrads[k].y += grads[k].y;
		}
	}
	for (int k = 0; k < K; ++k) {
		// Ugly: opencl compiler does not recognise the other double2 union fields(.s[i])
		theta[k].x = fabs(
				theta[k].x + eps_t
				* (eta.x - theta[k].x
						+ scale * ggrads[k].x)
				+ sqrt(2.0 * eps_t * theta[k].x) * noise[k].x);
		theta[k].y = fabs(
				theta[k].y + eps_t
				* (eta.y - theta[k].y
						+ scale * ggrads[k].y)
				+ sqrt(2.0 * eps_t * theta[k].y) * noise[k].y);
	}

}

