import sys
import numpy as np
import abc
import math

class Learner(object):
    """
    This is base class for all concrete learners, including MCMC sampler, variational
    inference,etc. 
    """
    def __init__(self, args, network, compatibility_mode):
        """
        initialize base learner parameters.
        """    
        self._compatibility_mode = compatibility_mode

        self._network = network
        # model priors
        if args.alpha == 0.0:
            self._alpha = 1.0 / args.K
        else:
            self._alpha = args.alpha
        self._eta = np.zeros(2)
        self._eta[0] = args.eta0
        self._eta[1] = args.eta1
        self._average_count = 1
        
        # parameters related to control model 
        self._K = args.K
        self._epsilon = args.epsilon
        
        # parameters related to network 
        self._N = network.get_num_nodes()
        
        # model parameters to learn
        self._beta = np.zeros(self._K)
        self._pi = np.zeros((self._N, self._K))
                
        # parameters related to sampling
        self._mini_batch_size = args.mini_batch_size
        if self._mini_batch_size < 1:
            self._mini_batch_size = self._N/2   # default option. 
        
        # ration between link edges and non-link edges
        self._link_ratio = network.get_num_linked_edges()/(self._N*(self._N-1)/2.0)
        # check the number of iterations. 
        self._step_count = 1
        # store perplexity for all the iterations
        self._ppxs_held_out = []
        self._ppxs_test = []
        
        self._max_iteration = args.max_iteration
        self.CONVERGENCE_THRESHOLD = 0.000000000001
        
        self.stepsize_switch = False

        self._ppx_for_heldout = np.zeros(network.get_held_out_size())

        sys.stdout.write("K %d N %d\n" % (self._K, self._N))
        sys.stdout.write("alpha %.f eta %.f,%.f epsilon %.f\n" % (self._alpha, self._eta[0], self._eta[1], self._epsilon))
        sys.stdout.write("mini_batch size parameter %d\n" % self._mini_batch_size)
        sys.stdout.write("compatibility mode %s\n" % str(self._compatibility_mode))

        
    @abc.abstractmethod
    def run(self):
        """
        Each concrete learner should implement this. It basically
        iterate the data sets, then update the model parameters, until
        convergence. The convergence can be measured by perplexity score. 
         
        We currently support four different learners:
            1. MCMC for batch learning
            2. MCMC for mini-batch training
            3. Variational inference for batch learning
            4. Stochastic variational inference
        """
    
    """
    def get_ppxs_held_out(self):
        return self._ppxs_held_out
    
    def get_ppxs_test(self):
        return self._ppxs_test
    """
    
    def set_max_iteration(self, max_iteration):
        self._max_iteration = max_iteration
    
    def _cal_perplexity_held_out(self):
        return self.__cal_perplexity(self._network.get_held_out_set())
   
    def _cal_perplexity_test(self):
        return self.__cal_perplexity(self._network.get_test_set())
        
    def _is_converged(self):
        n = len(self._ppxs_held_out)
        if n < 2:
            return False
        if abs((self._ppxs_held_out[n-1] - self._ppxs_held_out[n-2])/self._ppxs_held_out[n-2]) \
                                                    > self.CONVERGENCE_THRESHOLD:
            return False
        
        return True
    
    def save_model(self):
        f = open('pi.txt', 'wb')
        for i in range(0, self._N):
            f.write(str(i)+ ": " + str(self._pi[i]) +"\n")
        f.close()
        
        f = open('communities.txt', 'wb')
        commus = {}
        for k in range(0, self._K):
            commus[k] = []
        
        for i in range(0, self._N):
            m = max(self._pi[i])
            for j in range(0, self._K):
                if self._pi[i][j] == m:
                    commus[j].append(i)
        for i in range(0, self._K):
            f.write(str(commus[i]) + "\n")
        f.close()
        
        
    
    def __cal_perplexity(self, data):
        """
        calculate the perplexity for data.
        perplexity defines as exponential of negative average log likelihood. 
        formally:
            ppx = exp(-1/N * \sum){i}^{N}log p(y))
        
        we calculate average log likelihood for link and non-link separately, with the 
        purpose of weighting each part proportionally. (the reason is that we sample 
        the equal number of link edges and non-link edges for held out data and test data,
        which is not true representation of actual data set, which is extremely sparse. 
        """
        
        link_likelihood = 0.0
        non_link_likelihood = 0.0
        edge_likelihood = 0.0 
        link_count = 0
        non_link_count = 0
        
        key_list = list(data.keys())
        if self._compatibility_mode: # for compatibility w/ C++
            key_list.sort()
        idx = 0
        for edge in key_list:
            edge_likelihood = self.__cal_edge_likelihood(self._pi[edge[0]], self._pi[edge[1]], \
                                                       data[edge], self._beta)
            # sys.stdout.write("%s in? %s -> %.12f\n" % (str(edge), str(edge in self._network.get_linked_edges()), edge_likelihood))
            self._ppx_for_heldout[idx] = (self._ppx_for_heldout[idx] * (self._average_count - 1) + edge_likelihood) / self._average_count
            if edge in self._network.get_linked_edges():
                link_count += 1
                link_likelihood += math.log(self._ppx_for_heldout[idx])
                # link_likelihood += edge_likelihood
            else:
                non_link_count += 1
                # non_link_likelihood += edge_likelihood
                non_link_likelihood += math.log(self._ppx_for_heldout[idx])
            idx += 1
        # print "ratio " + str(self._link_ratio) + " count: link " + str(link_count) + " " + str(link_likelihood) + " non-link " + str(non_link_count) + " " + str(non_link_likelihood)
        
        # weight each part proportionally. 
        # avg_likelihood1 = self._link_ratio*(link_likelihood/link_count) + \
        #                    (1-self._link_ratio)*(non_link_likelihood/non_link_count) 
        
        # direct calculation. 
        avg_likelihood = 0.0
        if link_count + non_link_count != 0:
            avg_likelihood = (link_likelihood + non_link_likelihood)/(link_count+non_link_count)
        if True:
            avg_likelihood1 = self._link_ratio*(link_likelihood/link_count) + \
                               (1-self._link_ratio)*(non_link_likelihood/non_link_count) 
            print str(avg_likelihood) + " " + str(link_likelihood/link_count) + " " + str(link_count) + " " + \
                  str(non_link_likelihood/non_link_count) + " " +str(non_link_count)+" " + str(avg_likelihood1)

        self._average_count = self._average_count + 1
        sys.stdout.write("average_count is: %d " % self._average_count)

        return (-avg_likelihood)            
    
    
    def dump(self, a, n, name):
        sys.stdout.write("%s " % name)
        for k in range(0, n):
            sys.stdout.write("%.12f " % a[k])
        sys.stdout.write("\n")

    def __cal_edge_likelihood(self, pi_a, pi_b, y, beta):
        """
        calculate the log likelihood of edge :  p(y_ab | pi_a, pi_b, \beta)
        in order to calculate this, we need to sum over all the possible (z_ab, z_ba)
        such that:  p(y|*) = \sum_{z_ab,z_ba}^{} p(y, z_ab,z_ba|pi_a, pi_b, beta)
        but this calculation can be done in O(K), by using some trick.  
        """
        s = 0.0
        if y == 1:
            for k in range(0, self._K):
                s+= pi_a[k]*pi_b[k]*beta[k]
        else:
            sum = 0.0
            for k in range(0, self._K):
                s+= pi_a[k] * pi_b[k]*(1-beta[k])
                sum += pi_a[k] * pi_b[k]
            s += (1-sum)*(1-self._epsilon) 
        
        if s < 1e-30:
            s = 1e-30
        return s
        
        """
        prob = 0
        s = 0
        for k in range(0, self._K):
            if y == 0:
                prob += pi_a[k] * pi_b[k] * (1-beta[k])
            else:
                prob += pi_a[k] * pi_b[k] * beta[k]
            s += pi_a[k] * pi_b[k]
        
        if y == 0:
            prob += (1-s) * (1-self._epsilon)
        else:
            prob += (1-s) * self._epsilon
        if prob < 0:
            print "adsfadsfadsf"
        return math.log(prob)
        """
        
    
