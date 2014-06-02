import numpy as np
import opencl4py as cl

class ClSampler(object):
    
    def __init__(self, K, num_nodes, num_edges, link_map, batch_size, sample_size):
        sample_size += 1
        self.K = K
        self.batch_size = batch_size
        self.ctx = cl.Platforms().create_some_context()
        self.max_nodes_in_batch = 2*batch_size
        print self.ctx.devices[0].platform.name, ' | ', self.ctx.devices[0].name
        self.queue = self.ctx.create_queue(self.ctx.devices[0])
        self._init_graph(num_nodes, num_edges, link_map)
        # create grid for all num_nodes (not batch_size) so mapping of nodeId works directly
        self.np_sample_neighbor_nodes = np.empty((num_nodes, sample_size), dtype=np.int32)
        self.np_Zs = np.empty((num_nodes, self.K), dtype=np.float32)
        #
        self.cl_sample_neighbor_nodes = self.ctx.create_buffer(cl.CL_MEM_READ_WRITE, size=self.np_sample_neighbor_nodes.nbytes)
        self.cl_Zs = self.ctx.create_buffer(cl.CL_MEM_READ_WRITE, size=self.np_Zs.nbytes);
        self.cl_nodes = self.ctx.create_buffer(cl.CL_MEM_READ_ONLY, size=self.max_nodes_in_batch*4) # at most: 2 unique nodes per edge, 4-bytes each
        self.cl_pi = self.ctx.create_buffer(cl.CL_MEM_READ_WRITE, size=num_nodes*self.K*4)# (N, K) float
        self.cl_beta = self.ctx.create_buffer(cl.CL_MEM_READ_WRITE, size=self.K*4) # (K) float
        self.cl_p = self.ctx.create_buffer(cl.CL_MEM_READ_WRITE, size=self.max_nodes_in_batch*self.K*4) # at most: 2 unique nodes per edge, K*8-bytes each
        self.cl_bounds = self.ctx.create_buffer(cl.CL_MEM_READ_WRITE, size=self.max_nodes_in_batch*self.K*4)
        gopts = ['-IOpenCL/include']
        sampler_opts = gopts + ['-DK=%d' % self.K, '-DNEIGHBOR_SAMPLE_SIZE=%d' % (sample_size)]
        self.gprog = self.ctx.create_program(''.join(open('OpenCL/graph.cl', 'r').readlines()), options=' '.join(gopts))
        self.graph_init_kernel = self.gprog.get_kernel('graph_init')
        self.prog = self.ctx.create_program(''.join(open('OpenCL/sampler.cl', 'r').readlines()), options=' '.join(sampler_opts))
        self.sample_latent_vars_kernel = self.prog.get_kernel('sample_latent_vars')
        self.graph_init_kernel.set_arg(0, self.cl_graph)
        self.graph_init_kernel.set_arg(1, self.cl_edges)
        self.graph_init_kernel.set_arg(2, self.cl_node_edges)
        self.queue.execute_kernel(self.graph_init_kernel, (1,), (1,))
        self.queue.finish()
#        print self.prog.sample_latent_vars.get_work_group_info(cl.kernel_work_group_info.PRIVATE_MEM_SIZE, self.ctx.devices[0])

    def _init_graph(self, num_nodes, num_edges, link_map):
        self.np_edges = np.empty(num_edges*2, dtype=np.int32)
        self.np_node_edges = np.zeros( (num_nodes, 2) , dtype=np.int32) # (num links, offset) per node
        offset = 0
        for n, links in link_map.items():
            self.np_node_edges[n] = [len(links), offset]
            for d in links:
                self.np_edges[offset] = d
                offset += 1
        # create CL buffers for graph
        self.cl_graph = self.ctx.create_buffer(cl.CL_MEM_READ_WRITE, size=2*64/8) # 2 pointers, each is at most 64-bits
        self.cl_edges = self.ctx.create_buffer(cl.CL_MEM_READ_ONLY|cl.CL_MEM_COPY_HOST_PTR, host_array=self.np_edges)
        self.cl_node_edges = self.ctx.create_buffer(cl.CL_MEM_READ_ONLY|cl.CL_MEM_COPY_HOST_PTR, host_array=self.np_node_edges)

    def update_node_neighbors(self, nodeId, neighbor_nodes):
        self.np_sample_neighbor_nodes[nodeId] = neighbor_nodes

    def sample_latent_vars(self, nodes, pi, beta, epsilon):
        g_items = len(nodes)
        l_items = 32
        if g_items % l_items:
            g_items += l_items - (g_items % l_items)
        pi = pi.astype(np.float32)
        beta = beta.astype(np.float32)
        np_nodes = np.array(nodes, dtype=np.int32)
        self.queue.write_buffer(self.cl_nodes, np_nodes)
        self.queue.write_buffer(self.cl_pi, pi)
        self.queue.write_buffer(self.cl_beta, beta)
        self.queue.write_buffer(self.cl_sample_neighbor_nodes, self.np_sample_neighbor_nodes)
        self.sample_latent_vars_kernel.set_arg(0, self.cl_graph)
        self.sample_latent_vars_kernel.set_arg(1, self.cl_nodes)
        self.sample_latent_vars_kernel.set_arg(2, np.array([len(nodes)], dtype=np.int32)[0:1])
        self.sample_latent_vars_kernel.set_arg(3, self.cl_sample_neighbor_nodes)
        self.sample_latent_vars_kernel.set_arg(4, self.cl_pi)
        self.sample_latent_vars_kernel.set_arg(5, self.cl_beta)
        self.sample_latent_vars_kernel.set_arg(6, np.array([epsilon], dtype=np.float32)[0:1])
        self.sample_latent_vars_kernel.set_arg(7, self.cl_Zs)
        self.sample_latent_vars_kernel.set_arg(8, self.cl_p)
        self.sample_latent_vars_kernel.set_arg(9, self.cl_bounds)
        self.queue.execute_kernel(self.sample_latent_vars_kernel, (g_items,), (l_items,))
        self.queue.finish()
        self.queue.read_buffer(self.cl_Zs, self.np_Zs)
        return self.np_Zs
