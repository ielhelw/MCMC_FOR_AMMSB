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
		global int *Z;
		global double *Scratch;
		global ulong2 *RandomSeed;
		global int *errCtrl;
		global char *errMsg;
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
		global int *Z,
		global double *Scratch,
		global ulong2 *RandomSeed,
		global int *errCtrl,
		global char *errMsg
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
	bufs->bufs.Z = Z;
	bufs->bufs.Scratch = Scratch;
	bufs->bufs.RandomSeed = RandomSeed;
	bufs->bufs.errCtrl = errCtrl;
	bufs->bufs.errMsg = errMsg;
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

#define sample_z_ab_from_edge_expr_y_lifted(i) \
		(pi_a[i] * (pi_b[i] * y2_1 * ((beta[i] - epsilon) + epsilon) - y_1))

// #define sample_z_ab_from_edge_expr sample_z_ab_from_edge_expr_y_lifted
#define sample_z_ab_from_edge_expr sample_z_ab_from_edge_expr_optimized

inline int sample_z_ab_from_edge(
		global const double* pi_a,
		global const double *pi_b,
		global const double *beta,
		const double epsilon, const int y,
		const double random,
		global double *p
		) {
	int y_1 = y - 1;
	int y2_1 = y + y_1;
	p[0] = sample_z_ab_from_edge_expr(0);
	for (int i = 1; i < K; ++i) {
		p[i] = p[i-1] + sample_z_ab_from_edge_expr(i);
	}

	double location = random * p[K-1];
	return find_le(p, location, K, 0);
}

inline int sample_latent_vars_of(
		global Buffers *bufs,
		const int node,
		global const Graph *g,
		global const Graph *hg,
		global int* neighbor_nodes,
		global int* neighbor_nodes_hash,
		global const double *pi,
		global const double *beta,
		const double epsilon,
		global int *z, /* K elements */
		ulong2* randomSeed,
		global double *p) {
	for (int i = 0; i < K; ++i) z[i] = 0;

	for (int i = 0; i < NEIGHBOR_SAMPLE_SIZE; ++i) {
		int neighborId;
		int ret;
		for (;;) {
			neighborId = randint(randomSeed, 0, MAX_NODE_ID);
			const bool cond1 = neighborId != node;
			const bool cond2 = !graph_has_peer(hg, node, neighborId);
			const bool cond = cond1 && cond2;
			if (cond) {
				ret = hash_put(neighborId, neighbor_nodes_hash, HASH_SIZE);
				if (HASH_OK(ret)) {
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

	for (int i = 0; i < NEIGHBOR_SAMPLE_SIZE; ++i) {
		int neighborLoc = neighbor_nodes[i];
		int neighbor = neighbor_nodes_hash[neighborLoc];
		neighbor_nodes_hash[neighborLoc] = HASH_EMPTY; // reset the hash bucket to empty

		int y_ab = graph_has_peer(g, node, neighbor);
		int z_ab = sample_z_ab_from_edge(
				pi + node * K, pi + neighbor * K,
				beta, epsilon, y_ab,
				random(randomSeed),
				p);
		z[z_ab] += 1;
	}
	return 0;
}

void update_pi_for_node_(
		int node,
		global double *pi,// #K
		global double *phi,// #K
		global int *z, // #K
		ulong2 *randomSeed,
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
				+ sqrt(eps_t * phi[k]) * randn(randomSeed));
		phi[k] = phi_star_k;
	}
	double phi_sum = 0;
	for (int i = 0; i < K; ++i) phi_sum += phi[i];
	for (int i = 0; i < K; ++i) {
		pi[i] = phi[i]/phi_sum;
	}
}

kernel void sample_latent_vars(
		global Buffers *bufs,
		const int N, // #nodes
		const double epsilon
		){
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	global double *_p = bufs->bufs.Scratch + gid * K;
	ulong2 randomSeed = bufs->bufs.RandomSeed[gid];

	for (int i = gid; i < N; i += gsize) {
		int node = bufs->bufs.Nodes[i];
		int ret = sample_latent_vars_of(
				bufs,
				node,
				bufs->bufs.G, bufs->bufs.HG,
				// index by gid
				bufs->bufs.NodesNeighbors + gid * NEIGHBOR_SAMPLE_SIZE,
				bufs->bufs.NodesNeighborsHash + gid * HASH_SIZE,
				bufs->bufs.Pi,
				bufs->bufs.Beta,
				epsilon,
				bufs->bufs.Z + i * K,
				&randomSeed,
				_p);
		if (ret) break;
	}
	bufs->bufs.RandomSeed[gid] = randomSeed;
}

kernel void update_pi(
		global Buffers *bufs,
		const int N, // #nodes
		double alpha,
		double a, double b, double c,
		int step_count, int total_node_count
		){
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	global double *_p = bufs->bufs.Scratch + gid * K;
	ulong2 randomSeed = bufs->bufs.RandomSeed[gid];
	for (int i = gid; i < N; i += gsize) {
		int node = bufs->bufs.Nodes[i];
		update_pi_for_node_(node,
				bufs->bufs.Pi + node * K,
				bufs->bufs.Phi + node * K,
				bufs->bufs.Z + i * K,
				&randomSeed,
				_p,
				alpha, a, b, c, step_count, total_node_count);
	}
	bufs->bufs.RandomSeed[gid] = randomSeed;
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
		global Buffers *bufs,
		int E // #edges
		) {
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	ulong2 randomSeed = bufs->bufs.RandomSeed[gid];

	for (int i = gid; i < E; i += gsize) {
		bufs->bufs.Z[i] = sample_latent_vars2_(
				bufs->bufs.Edges[i],
				bufs->bufs.G,
				bufs->bufs.Pi,
				bufs->bufs.Beta,
				bufs->bufs.Scratch + gid * (K+1),
				random(&randomSeed));
	}
	bufs->bufs.RandomSeed[gid] = randomSeed;
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
	}
}

kernel void update_beta_calculate_grads(
		global Buffers *bufs,
		const int E, // #edges
		double scale,
		const int count_partial_sums
		) {
	size_t gid = get_global_id(0);
	if (gid < count_partial_sums) {
		size_t gsize = get_global_size(0);
		global double2 *grads = (global double2 *)bufs->bufs.Scratch + gid * K;

		for (int i = 0; i < K; ++i) {
			grads[i].x = 0;
			grads[i].y = 0;
		}
		for (int i = gid; i < E; i += gsize) {
			int y_ab = graph_has_peer(bufs->bufs.G, bufs->bufs.Edges[i].x, bufs->bufs.Edges[i].y);
			int k = bufs->bufs.Z[i];
			if (k != -1) {
				grads[k].x += (1-y_ab) / bufs->bufs.Theta[k].x - 1 / bufs->bufs.ThetaSum[k];
				grads[k].y += y_ab / bufs->bufs.Theta[k].y - 1 / bufs->bufs.ThetaSum[k];
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
	for (int i = 1; i < count_partial_sums; ++i) {
		global double2 *grads = (global double2 *)bufs->bufs.Scratch + i * K;
		for (int k = 0; k < K; ++k) {
			ggrads[k].x += grads[k].x;
			ggrads[k].y += grads[k].y;
		}
	}
	for (int k = 0; k < K; ++k) {
		// Ugly: opencl compiler does not recognise the other double2 union fields(.s[i])
		bufs->bufs.Theta[k].x = fabs(
				bufs->bufs.Theta[k].x + eps_t
				* (eta.x - bufs->bufs.Theta[k].x
						+ scale * ggrads[k].x)
				+ sqrt(2.0 * eps_t * bufs->bufs.Theta[k].x) * randn(&randomSeed));
		bufs->bufs.Theta[k].y = fabs(
				bufs->bufs.Theta[k].y + eps_t
				* (eta.y - bufs->bufs.Theta[k].y
						+ scale * ggrads[k].y)
				+ sqrt(2.0 * eps_t * bufs->bufs.Theta[k].y) * randn(&randomSeed));
	}
	bufs->bufs.RandomSeed[gid] = randomSeed;
}

// Expand the python fully, knowing this is two dimensions so we don't need
// to synchronize to pre-calculate the sum:
// 		temp = self.__theta/np.sum(self.__theta,1)[:,np.newaxis]
// 		self._beta = temp[:,1]
kernel void update_beta_calculate_beta(
		global double2 *theta,		// #K
		global double *beta			// #K
		)
{
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	for (int k = gid; k < K; k += gsize) {
		beta[k] = theta[k].y / (theta[k].x + theta[k].y);
	}
}
