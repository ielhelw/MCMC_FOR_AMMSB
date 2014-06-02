#include "graph.h"

#ifndef K
#error "Need definition of K"
#endif

#ifndef NEIGHBOR_SAMPLE_SIZE
#error "Need definition of NEIGHBOR_SAMPLE_SIZE"
#endif

#if 0

// adapted from sample_latent_vars.pyx
int sample_z_ab_from_edge(
		global float* pi_a,
		global float *pi_b,
		global float *beta,
		float epsilon, float y,
		global float *p,
		global float *bounds) {
	float location = 0;
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
		global float* pi_a,
		global float *pi_b,
		global float *beta,
		float epsilon, float y,
		global float *p,
		global float *bounds) {
	float location = 0;
	float bound = pow(beta[0], y) * pow(1-beta[0], 1-y) * pi_a[0] * pi_b[0]
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
		global float *pi,
		global float *beta,
		float epsilon,
		global float *z /* K elements */,
		global float *p,
		global float *bounds) {
	for (int i = 0; i < K; ++i) z[i] = 0;
	for (int i = 0; i < NEIGHBOR_SAMPLE_SIZE; ++i) {
		int neighbor = neighbor_nodes[i];
		int y_ab = graph_has_peer(g, node, neighbor);
		int z_ab = sample_z_ab_from_edge(
				pi + node * K, pi + neighbor * K,
				beta, epsilon, y_ab, p, bounds);
		z[z_ab] += 1;
	}
}

kernel void sample_latent_vars(
		global Graph *g,
		global int *nodes,
		int N, // #nodes
		global int *neighbor_nodes,// (#total_nodes, NEIGHBOR_SAMPLE_SIZE)
		global float *pi,// (#total_nodes, K)
		global float *beta,// (#K)
		float epsilon,
		global float *Z,// (#total_nodes, K)
		global float *p,// (#nodes, K)
		global float *bounds// (#nodes, K)
) {
	size_t gid = get_global_id(0);
	size_t gsize = get_global_size(0);
	global float *_p = p + gid * K;
	global float *_b = bounds + gid * K;

	for (int i = gid; i < N; i += gsize) {
		int node = nodes[i];
		sample_latent_vars_of(
				node,
				g,
				neighbor_nodes + node * NEIGHBOR_SAMPLE_SIZE,
				pi,
				beta,
				epsilon,
				Z + nodes[i] * K,
				_p, _b);
	}
}

void update_pi_for_node_(
		int node,
		global float *pi,// #K
		global float *phi,// #K
		global float *z, // #K
		global float *noise, // #K
		global float *grad, // #K
		float alpha,
		float a, float b, float c,
		float step_count, int total_node_count
		) {
	float eps_t = a * pow((1 + step_count/b), -c);
	float phi_i_sum = 0;
	for (int i = 0; i < K; ++i) phi_i_sum += phi[i];
	for (int k = 0; k < K; ++k) {
		grad[k] = -NEIGHBOR_SAMPLE_SIZE * 1/phi_i_sum;
		grad[k] += 1/phi[k] * z[k];
	}
	for (int k = 0; k < K; ++k) {
		float phi_star_k = fabs(phi[k] + eps_t/2
				* (alpha - phi[k] + total_node_count/NEIGHBOR_SAMPLE_SIZE * grad[k])
				+ pow(eps_t, 0.5f) * pow(phi[k], 0.5f) * noise[k]);
		phi[k] = phi_star_k * (1.0f/step_count) + (1-1.0f/step_count) * phi[k];
	}
	float phi_sum = 0;
	for (int i = 0; i < K; ++i) phi_sum += phi[i];
	for (int i = 0; i < K; ++i) {
		pi[i] = phi[i]/phi_sum;
	}
}

kernel void update_pi_for_node(
		global int *nodes,
		int N, // #nodes
		global float *pi,// (#total_nodes, K)
		global float *phi,// (#total_nodes, K)
		global float *Z, // (#total_nodes, K)
		global float *noise, // (#nodes, K)
		global float *grad, // (#nodes, K)
		float alpha,
		float a, float b, float c,
		float step_count, int total_node_count
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
