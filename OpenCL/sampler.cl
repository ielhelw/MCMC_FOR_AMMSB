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

#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define NODE_ID_VALID(NID) ((NID) >= 0 && (NID) < MAX_NODE_ID)

#define MCMC_CL_STOCHASTIC_USE_SCRATCH

#if 1

// adapted from sample_latent_vars.pyx
int sample_z_ab_from_edge(
		global double* pi_a,
		global double *pi_b,
		global double *beta,
		double epsilon, int y,
		global double *p,
		global double *bounds) {
	double location = 0;
	for (int i = 0; i < K; ++i) {
		p[i] = pow(beta[i], y) * pow(1-beta[i], 1-y) * pi_a[i] * pi_b[i]
		+ pow(epsilon, y) * pow(1-epsilon, 1-y) * pi_a[i] * (1-pi_b[i]);
	}
	bounds[0] = p[0];
	for (int i = 1; i < K; ++i) {
		bounds[i] = bounds[i-1] + p[i];
	}
	for (int i = 0; i < K; ++i) {
		if (location <= bounds[i]) return i;
	}
	return -1;
}

#else

// merged version
int sample_z_ab_from_edge(
		global double* pi_a,
		global double *pi_b,
		global double *beta,
		double epsilon, int y
#ifdef MCMC_CL_STOCHASTIC_USE_SCRATCH
		,
		global double *p,
		global double *bounds
#endif
		) {
	double location = 0;
	double bound = pow(beta[0], y) * pow(1-beta[0], 1-y) * pi_a[0] * pi_b[0]
		+ pow(epsilon, y) * pow(1-epsilon, 1-y) * pi_a[0] * (1-pi_b[0]);
	if (location <= bound) return 0;
	for (int i = 1; i < K; ++i) {
		bound += pow(beta[i], y) * pow(1-beta[i], 1-y) * pi_a[i] * pi_b[i]
		+ pow(epsilon, y) * pow(1-epsilon, 1-y) * pi_a[i] * (1-pi_b[i]);
		if (location <= bound) return i;
	}
	return -1;
}

#endif

void sample_latent_vars_of(
		int node,
		global Graph *g,
		global int* neighbor_nodes,
		global double *pi,
		global double *beta,
		double epsilon,
		global double *z /* K elements */
#ifdef MCMC_CL_STOCHASTIC_USE_SCRATCH
		,
		global double *p,
		global double *bounds
#endif
		) {
	for (int i = 0; i < K; ++i) z[i] = 0;
	for (int i = 0; i < NEIGHBOR_SAMPLE_SIZE; ++i) {
		int neighbor = neighbor_nodes[i];

//		printf("%d: (%d, %d)\n", i, node, neighbor);

		int y_ab = graph_has_peer(g, node, neighbor);
		int z_ab = sample_z_ab_from_edge(
				pi + node * K, pi + neighbor * K,
				beta, epsilon, y_ab
#ifdef MCMC_CL_STOCHASTIC_USE_SCRATCH
				, p, bounds
#endif
				);
		z[z_ab] += 1;
	}
}

kernel void sample_latent_vars(
		global Graph *g,
		global int *nodes,
		int N, // #nodes
		global int *neighbor_nodes,// (#total_nodes, NEIGHBOR_SAMPLE_SIZE)
		global double *pi,// (#total_nodes, K)
		global double *beta,// (#K)
		double epsilon,
		global double *Z /* (#total_nodes, K) */
#ifdef MCMC_CL_STOCHASTIC_USE_SCRATCH
		,
		global double *p,// (#nodes, K)
		global double *bounds// (#nodes, K)
#endif
) {
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
#ifdef MCMC_CL_STOCHASTIC_USE_SCRATCH
	global double *_p = p + gid * K;
	global double *_b = bounds + gid * K;
#endif

	for (int i = gid; i < N; i += gsize) {
		int node = nodes[i];
		/**/
		global int* neighbor = neighbor_nodes + node * NEIGHBOR_SAMPLE_SIZE;
		for (int j = 0; j < NEIGHBOR_SAMPLE_SIZE; ++j) {
			if (!NODE_ID_VALID(neighbor[j])) {
				printf("========================================= %d: %d -> %d\n", i, node, neighbor[j]);
			}
		}
		/**/
		sample_latent_vars_of(
				node,
				g,
				neighbor_nodes + node * NEIGHBOR_SAMPLE_SIZE,
				pi,
				beta,
				epsilon,
				Z + nodes[i] * K
#ifdef MCMC_CL_STOCHASTIC_USE_SCRATCH
				, _p, _b
#endif
				);
	}
}

void update_pi_for_node_(
		int node,
		global double *pi,// #K
		global double *phi,// #K
		global double *z, // #K
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
				* (alpha - phi[k] + total_node_count/NEIGHBOR_SAMPLE_SIZE * grad[k])
				+ pow(eps_t, 0.5) * pow(phi[k], 0.5) * noise[k]);
		phi[k] = phi_star_k * (1.0/step_count)
				+ (1-1.0/step_count) * phi[k];
	}
	double phi_sum = 0;
	for (int i = 0; i < K; ++i) phi_sum += phi[i];
	for (int i = 0; i < K; ++i) {
		pi[i] = phi[i]/phi_sum;
	}
}

kernel void update_pi_for_node(
		global int *nodes,
		int N, // #nodes
		global double *pi,// (#total_nodes, K)
		global double *phi,// (#total_nodes, K)
		global double *Z, // (#total_nodes, K)
		global double *noise, // (#nodes, K)
		global double *grad, // (#nodes, K)
		double alpha,
		double a, double b, double c,
		int step_count, int total_node_count
		) {
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	for (int i = gid; i < N; i += gsize) {
		update_pi_for_node_(nodes[i],
				pi + nodes[i] * K,
				phi + nodes[i] * K,
				Z + nodes[i] * K,
				noise + i * K,
				grad + i * K,
				alpha, a, b, c, step_count, total_node_count);
	}
}

int sample_latent_vars2_(
		int2 edge,
		global Graph *g,
		global double *pi,// (#total_nodes, K)
		global double *beta,// (#K)
		global double *p,// #K+1
		global double *bounds,// #K+1
		double r
		) {
	int y = graph_has_peer(g, edge.x, edge.y);
	global double *pi_a = pi + edge.x * K;
	global double *pi_b = pi + edge.y * K;
	double p_sum = 0;
	for (int k = 0; k < K; ++k) {
		p[k] = pow(beta[k], y) * pow(1-beta[k], 1-y) * pi_a[k] * pi_b[k];
		p_sum += p[k];
	}
	p[K] = 1 - p_sum;
	bounds[0] = p[0];
	for (int k = 1; k < K+1; ++k) {
		bounds[k] = bounds[k-1] + p[k];
	}
	double location = r * bounds[K];
	for (int i = 0; i < K; ++i) {
		if (location <= bounds[i]) return i;
	}
	return -1;
}

kernel void sample_latent_vars2(
		global Graph *g,
		global int2 *edges,
		int E, // #edges
		global double *pi,// (#total_nodes, K)
		global double *beta,// (#K)
		global int *Z,// #edges
		global double *p,// (#edges, K+1)
		global double *bounds,// (#edges, K+1)
		global double *r
		) {
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	for (int i = gid; i < E; i += gsize) {
		Z[i] = sample_latent_vars2_(edges[i],
				g,
				pi,
				beta,
				p + i * (K+1),
				bounds+ i * (K+1),
				r[i]);
	}
}

