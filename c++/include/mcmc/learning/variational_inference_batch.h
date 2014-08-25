#ifndef MCMC_LEARNING_VARIATIONAL_INFERENCE_BATCH_H__
#define MCMC_LEARNING_VARIATIONAL_INFERENCE_BATCH_H__

#include "mcmc/learning/learner.h"
#include "mcmc/estimate_phi.h"


namespace mcmc {
namespace learning {

// Helper classes for the transforms
class UpdateGammaStar : public binary_function<double, double, double> {
	UpdateGammaStar(double p_t, double alpha) : p_t(p_t), alpha(alpha) {
	}

	double operator()(const double &x, const double &y) {
		// gamma_star = (1-p_t)*gamma[node] + p_t * (alpha + gamma_grad[node]);
		return (1.0 - p_t) * x + p_t * (alpha + y);
	}

protected:
	double p_t;
	double alpha;
};


class UpdateGamma : public binary_function<double, double, double> {
	UpdateGamma(::size_t step_count) : step_count(step_count) {
	}

	double operator()(const double &x, const double &y) {
		// gamma[node] = (1-1.0/(step_count))*gamma[node] + gamma_star;
		return (1.0 - 1.0 / step_count) * x + y;
	}

protected:
	::size_t step_count;
};


UpdateGamma update_gamma(step_count);

class SV : public Learner {
public:

	/**
	 * Initialize the sampler using the network object and arguments (i.e prior)
	 * Arguments:
	 *     network:    representation of the graph.
	 *     args:       containing priors, control parameters for the model.
	 */
	SV(const Options &args, const Network &graph) : Learner(args, graph) {
		// variational parameters.
		lamda = Random::random.gamma(eta[0], eta[1], K, 2);	// variational parameters for beta
		gamma = Random::random.gamma(1, 1, N, K);			// variational parameters for pi
		std::cerr << "gamma.size() " << gamma.size() << " gamma[0].size() " << gamma[0].size() << std::endl;
		update_pi_beta();
		// step size parameters.
		kappa = args.b;
		tao = args.c;

		// control parameters for learning
		online_iterations = 50;
		phi_update_threshold = 0.0001;

		// lift
		log_epsilon = log(epsilon);
		log_1_epsilon = log(1.0 - epsilon);
	}


	virtual ~SV() {
	}


	virtual void run() {
        // pr = cProfile.Profile()
        // pr.enable()

        // running until convergence.
        step_count++;

		while (step_count < max_iteration and !is_converged()) {
            double ppx_score = cal_perplexity_held_out();
			std::cout << "perplexity for hold out set is: "  << ppx_score << std::endl;

            update();
            update_pi_beta();

            step_count++;
		}

        // pr.disable()
#if 0
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
#endif
	}


protected:
    void update() {
		std::vector<std::vector<double> > gamma_grad(N, std::vector<double>(K, 0.0));
		std::vector<std::vector<double> > lamda_grad(K, std::vector<double>(2, 0.0));

		for (::size_t a = 0; a < N; a++) {
			for (::size_t b = a + 1; b < N; b++) {
				Edge edge(a, b);
				if (network.get_held_out_set()->find(edge) != network.get_held_out_set()->end() ||
						network.get_test_set()->find(edge) != network.get_test_set()->end()) {
                    continue;
				}

				DoubleVectorPair phi_abba = sample_latent_vars_for_each_pair(a, b, gamma[a], gamma[b],
																			 lamda, K, phi_update_threshold,
																			 epsilon, online_iterations,
																			 network.get_linked_edges());

                // update gamma_grad and lamda_grad
				std::transform(gamma_grad[a].begin(), gamma_grad[a].end(),
							   phi_abba.first->begin(), phi_abba.first->end(),
							   gamma_grad[a].begin());
				std::transform(gamma_grad[b].begin(), gamma_grad[b].end(),
							   phi_abba.second->begin(), phi_abba.second->end(),
							   gamma_grad[b].begin());

                int y = 0;
				if (network.get_linked_edges()->find(edge) != network.get_linked_edges()->end()) {
                    y = 1;
				}

                for (::size_t k = 0; k < K; k++) {
                    lamda_grad[k][0] += phi_abba->first[k] * phi_abba->second[k] * y;
                    lamda_grad[k][1] += phi_abba->first[k] * phi_abba->second[k] * (1-y);
				}
				std::cerr << "GC phi_abba vectors?" << std::endl;
			}
		}

        // update gamma, only update node in the grad
		double p_t;
        if (! self.stepsize_switch) {
            p_t = std::pow(1024 + step_count, -0.5);
		} else {
            p_t = 0.01*std::pow(1+step_count/1024.0, 0.55);
		}

		UpdateGammaStar update_gamma_star(p_t, alpha);
		UpdateGamma update_gamma(step_count);
        for (::size_t node = 0; node < K; node++) {
			std::vector<double> gamma_star(K, 0.0);

            if (step_count > 400) {
                // gamma_star = (1-p_t)*gamma[node] + p_t * (alpha + gamma_grad[node]);
				std::transform(gamma[node].begin(), gamma[node].end(),
							   gamma_grad[node].begin(),
							   gamma_star.begin(),
							   update_gamma_star);
                // gamma[node] = (1-1.0/(step_count))*gamma[node] + gamma_star;
				std::transform(gamma[node].begin(), gamma[node].end(),
							   gamma_star[node].begin(),
							   gamma[node].begin(),
							   update_gamma);
			} else {
                // gamma[node]=(1-p_t)*gamma[node] + p_t * (alpha + gamma_grad[node]);
				std::transform(gamma[node].begin(), gamma[node].end(),
							   gamma_grad[node].begin(),
							   gamma_star.begin(),
							   update_gamma_star);
			}
		}

        // update lamda
        for (::size_t k = 0; k < K; k++) {

            if (step_count > 400) {
                double lamda_star_0 = (1.0-p_t)*lamda[k][0] + p_t *(eta[0] + lamda_grad[k][0]);
                double lamda_star_1 = (1.0-p_t)*lamda[k][1] + p_t *(eta[1] + lamda_grad[k][1]);
                lamda[k][0] = (1.0-1.0/(step_count)) * lamda[k][0] +1.0/(step_count)*lamda_star_0;
                lamda[k][1] = (1.0-1.0/(step_count)) * lamda[k][1] +1.0/(step_count)*lamda_star_1;
			} else {
                lamda[k][0] = (1.0-p_t)*lamda[k][0] + p_t *(eta[0] + lamda_grad[k][0]);
                lamda[k][1] = (1.0-p_t)*lamda[k][1] + p_t *(eta[1] + lamda_grad[k][1]);
			}
		}
	}


    void update_pi_beta() {
#ifdef UNUSED
        pi = gamma/np.sum(gamma,1)[:,np.newaxis]
        temp = lamda/np.sum(lamda,1)[:,np.newaxis]
        beta = temp[:,1]
#endif
	}


#ifdef UNUSED
    def __estimate_phi_for_edge(self, edge, phi):
        '''
        calculate (phi_ab, phi_ba) for given edge : (a,b)
        (a) calculate phi_ab given phi_ba
            if y =0:
        phi_ab[k]=exp(psi(gamma[a][k])+phi_ba[k]*(psi(lamda[k][1])-psi(lambda[k0]+lambda[k][1]))-phi_ba[k]*log(1-epsilon))
            if y=1:
        phi_ab[k]=exp(psi(gamma[a][k])+phi_ba[k]*(psi(lamda[k][0])-psi(lambda[k0]+lambda[k][1]))-phi_ba[k]*log(epsilon))

        (b) calculate phi_ba given phi_ab
            if y =0:
        phi_ba[k]=exp(psi(gamma[b][k])+phi_ab[k]*(psi(lamda[k][1])-psi(lambda[k0]+lambda[k][1]))-phi_ab[k]*log(1-epsilon))
            if y=1:
        phi_ba[k]=exp(psi(gamma[b][k])+phi_ab[k]*(psi(lamda[k][0])-psi(lambda[k0]+lambda[k][1]))-phi_ab[k]*log(epsilon))

        '''

        a = edge[0]
        b = edge[1]
        // initialize
        phi_ab = np.empty(K)
        phi_ba = np.empty(K)
        phi_ab.fill(1.0/K)
        phi_ba.fill(1.0/K)

        y = 0
        if (a,b) in network.get_linked_edges():
            y = 1

        // alternatively update phi_ab and phi_ba, until it converges
        // or reach the maximum iterations.
        for i in range(online_iterations):
            phi_ab_old = copy.copy(phi_ab)
            phi_ba_old = copy.copy(phi_ba)

            // first, update phi_ab
            for k in range(K):
                if y == 1:
                    u = -phi_ba[k]* math.log(epsilon)
                    phi_ab[k] = math.exp(psi(gamma[a][k])+phi_ba[k]*\
                                         (psi(lamda[k][0])-psi(lamda[k][0]+lamda[k][1]))+u)
                else:
                    u = -phi_ba[k]* math.log(1-epsilon)
                    phi_ab[k] = math.exp(psi(gamma[a][k])+phi_ba[k]*\
                                         (psi(lamda[k][1])-psi(lamda[k][0]+lamda[k][1]))+u)
            sum_phi_ab = np.sum(phi_ab)
            phi_ab = phi_ab/sum_phi_ab

            // then update phi_ba
            for k in range(K):
                if y == 1:
                    u = -phi_ab[k]* math.log(epsilon)
                    phi_ba[k] = math.exp(psi(gamma[b][k])+phi_ab[k]*\
                                         (psi(lamda[k][0])-psi(lamda[k][0]+lamda[k][1]))+u)
                else:
                    u = -phi_ab[k]* math.log(1-epsilon)
                    phi_ba[k] = math.exp(psi(gamma[b][k])+phi_ab[k]*\
                                         (psi(lamda[k][1])-psi(lamda[k][0]+lamda[k][1]))+u)

            sum_phi_ba = np.sum(phi_ba)
            phi_ba = phi_ba/sum_phi_ba


            // calculate the absolute difference between new value and old value
            diff1 = np.sum(np.abs(phi_ab - phi_ab_old))
            diff2 = np.sum(np.abs(phi_ba - phi_ba_old))
            if diff1 < phi_update_threshold and diff2 < phi_update_threshold:
                break

        phi[(a,b)] = phi_ab
        phi[(b,a)] = phi_ba
#endif

};

}	// namespace learning
}	// namespace mcmc

#endif	// ndef MCMC_LEARNING_VARIATIONAL_INFERENCE_BATCH_H__