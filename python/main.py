from com.uva.source_aware_random import SourceAwareRandom as random
import argparse
from com.uva.network import Network
from com.uva.preprocess.data_factory import DataFactory
from com.uva.learning.mcmc_sampler_stochastic import MCMCSamplerStochastic
from com.uva.timer import Timer

import threading
import sys

def work_mcmc (sampler, ppxs): 
    threading.Timer(2, work_mcmc, [sampler, ppxs]).start (); 
    ppx = sampler._cal_perplexity_held_out()
    print "MCMC perplexity: " + str(ppx)
    ppxs.append(ppx)
    if len(ppxs) % 100 == 0:
        f = open('result_mcmc.txt', 'wb')
        for i in range(len(ppxs)):
            f.write(str(ppxs[i]) + "\n")
        f.close()

def main():
    # default args for WenZhe's C++ implementation:
    #   --K=15 --alpha=0.01 --epsilon=0.0000001 --hold_out_prob=0.009999999776

    sys.stdout.write("Invoked with: ")
    for a in sys.argv:
        sys.stdout.write("%s " % a)
    sys.stdout.write("\n")

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', type=str, default="netscience.txt", required=False)
    parser.add_argument('--alpha', type=float, default=0.0, required=False)
    parser.add_argument('--eta0', type=float, default=1, required=False)
    parser.add_argument('--eta1', type=float, default=1, required=False)
    parser.add_argument('--K', '-K', type=int, default=300, required=False)
    parser.add_argument('--mini_batch_size', '-m', type=int, default=50, required=False)   # mini-batch size
    parser.add_argument('--num-node-sample', '-n', type=int, default=50, required=False)   # neighbor sample size
    parser.add_argument('--epsilon', type=float, default=1.0e-07, required=False)
    parser.add_argument('--max_iteration', '-x', type=int, default=10000000, required=False)
    parser.add_argument('--interval', '-i', type=int, default=1000, required=False)
    
    # parameters for step size
    parser.add_argument('--a', '-a', type=float, default=0.01, required=False)
    parser.add_argument('--b', '-b', type=float, default=1024.0, required=False)
    parser.add_argument('--c', '-c', type=float, default=0.55, required=False)
    
    parser.add_argument('--num_updates', type=int, default=1000, required=False)
    parser.add_argument('--hold_out_prob', type=float, default=0.1, required=False)
    parser.add_argument('--output_dir', type=str,default='.', required=False)
    args = parser.parse_args()

    # compatibility_mode = True
    compatibility_mode = False

    random.seed(42, compatibility_mode)

    # data = DataFactory.get_data("netscience")
    data = DataFactory.get_data("relativity", args.filename)
    network = Network(data, args.hold_out_prob, compatibility_mode)
    args.num_pieces = (network.get_num_nodes() + args.mini_batch_size - 1) / args.mini_batch_size
    # network.set_num_pieces(args.num_pieces)  # compatible w/ C++ implementation
        
    print "start MCMC stochastic"
    ppx_mcmc = []
    sampler = MCMCSamplerStochastic(args, network, compatibility_mode)
    sampler.set_num_node_sample(args.num_node_sample)     # compatible w/ C++ implementation
    #work_mcmc(sampler, ppx_mcmc)
    sampler.run()
        
if __name__ == '__main__':
    main()
