from time import time
from collections import defaultdict
import networkx as nx
import numpy as np
from scipy.special import logsumexp
from community import community_louvain
import scipy.stats as ss

def calc_decay(decay_type, d_ij, a):
    if d_ij < 0: return 0
    if decay_type == 'exponential':
        return float(np.exp(-d_ij / a))
    elif decay_type == 'window':
        return 1 if d_ij < a else 0
    elif decay_type == 'logistic':
        b = np.exp(-d_ij + a)
        return float(b / (1 + b))
    elif decay_type == 'crp':
        return 1


def get_linked(i, link):
    ac = set()
    q = [i]
    while q:
        cur = q[0]
        ac.update([cur])
        link_pos = np.where(link == cur)[0]
        link_pos = set(link_pos) - ac - set(q)
        q += list(link_pos)
        q = q[1:]
    ac = list(ac)
    return ac


class DynamicClusters:

    def __init__(self, pairs, times, args, true_links=None, true_clusters=None, alpha_params=None, gamma_params=None, tau_params=None, sigma_params=None,f1params=None, f2params=None):
        if alpha_params is None:
            self.alpha_params = np.ones(2)
        else:
            self.alpha_params = alpha_params
        if gamma_params is None:
            self.gamma_params = np.ones(2)
        else:
            self.gamma_params = gamma_params
        if tau_params is None:
            self.tau_params = np.ones(2)
        else:
            self.tau_params = tau_params
        if sigma_params is None:
            self.sigma_params = np.ones(2)
        else:
            self.sigma_params = sigma_params
        if f1params is None:
            self.f1params = np.array([50, 1])
        else:
            self.f1params = f1params
        if f2params is None:
            self.f2params = np.array([50, 1])
        else:
            self.f2params = f2params

        self.pairs = pairs
        self.num_pairs = self.pairs.shape[0]
        self.times = times
        self.true_links = true_links
        self.true_clusters = true_clusters
        self.n_iter = args.epochs
        self.sample_hypers = args.sample_hypers
        self.f1_decay = args.f1
        self.f1_param = args.f1_param
        self.f2_decay = args.f2
        self.f2_param = args.f2_param
        self.decay_scale = args.method_scale
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.tau = args.tau
        self.sigma = args.sigma
        if self.f1_decay == 'crp':
            self.sigma = 0.
        self.epsilon = 1e-10

        self.link = np.zeros(self.num_pairs).astype(int)
        self.cluster = np.zeros(self.num_pairs).astype(int)
        self.num_tables = 0
        self.index_node_map = {i: n for i, n in enumerate(np.unique(self.pairs))}
        self.node_index_map = defaultdict(lambda: -1)
        self.node_index_map.update({n: i for i, n in self.index_node_map.items()})
        self.num_users = len(self.index_node_map)
        self.tables_sigma = np.zeros(self.num_users+1)
        self.beta = np.zeros(self.num_users+1)

    def initialize_clusters_best_community(self, Ztrain):
        '''
        in case of initilizing cluster assignments, we use best_partition method of Python's library
        this partitions the vertices. We use the partioned vertices and partition their edges based on highest rates in a partition
        '''
        G = nx.Graph()
        nodes = list(np.unique(Ztrain))
        G.add_nodes_from(nodes)
        G.add_edges_from(Ztrain)
        partitions = community_louvain.best_partition(G)
        to_remove = []
        for n in partitions.keys():
            if n not in nodes:
                to_remove.append(n)
        for n in to_remove:
            partitions.pop(n)
        cluster_edges = defaultdict(lambda: [])
        for i in range(Ztrain.shape[0]):
            (u, v) = Ztrain[i]
            if u in partitions.keys():
                self.cluster[i] = partitions[u]
            elif v in partitions.keys():
                self.cluster[i] = partitions[v]
            else:
                self.cluster[i] = 0
            cluster_edges[self.cluster[i]].append(i)

        for c, edges in cluster_edges.items():
            edges = np.array(edges)
            self.cluster[edges] = edges[0]

    def get_log_priors(self,num_pairs, timestamps, f1=True, f2=True,f1param=None, f2param=None):
        if f1param is None:
            f1param = self.f1_param
        if f2param is None:
            f2param = self.f2_param

        log_prior_clusters, prior_tables = None, None

        if f1:
            log_prior_clusters = np.zeros((num_pairs, num_pairs))
        if f2:
            prior_tables = np.zeros((num_pairs, num_pairs))

        get_f1 = lambda d: calc_decay(self.f1_decay, d, f1param)
        get_f2 = lambda d: calc_decay(self.f2_decay, d, f2param)
        for i in range(num_pairs):
            for j in range(num_pairs):
                dij = timestamps[i] - timestamps[j]
                if i == j:
                    if f1:
                        log_prior_clusters[i][i] = np.log(self.alpha)
                    if f2:
                        prior_tables[i][i] = 1
                elif i<j and f1:
                    log_prior_clusters[i][j] = -np.inf
                else:
                    if f1:
                        log_prior_clusters[i][j] = np.log(get_f1(dij))
                    if f2:
                        prior_tables[i][j] = get_f2(dij)

        return log_prior_clusters, prior_tables

    def cluster_likelihood_dynamic(self, cluster_inds, taubeta, priortables=None):
        cluster_size = cluster_inds.shape[0]
        if cluster_size == 0: return -np.inf

        cluster_pairs = self.pairs[cluster_inds]
        cluster_senders = cluster_pairs[:, 0]
        cluster_receivers = cluster_pairs[:, 1]

        if priortables is None:
            priors = self.prior_tables[cluster_inds, :][:, cluster_inds]
        else:
            priors = priortables[cluster_inds, :][:, cluster_inds]
        priors[np.triu_indices(cluster_size)] = 0
        normalize_factor = np.sum(priors, axis=1) + np.sum(taubeta)

        tbeta_s = np.zeros(cluster_size)
        tbeta_r = np.zeros(cluster_size)
        sen_mtx = np.zeros((cluster_size, cluster_size))
        rec_mtx = np.zeros((cluster_size, cluster_size))
        for idx in range(cluster_size):
            sen_mtx[idx][cluster_senders == cluster_senders[idx]] = 1
            rec_mtx[idx][cluster_receivers == cluster_receivers[idx]] = 1
            tbeta_s[idx] = taubeta[self.node_index_map[cluster_senders[idx]]]
            tbeta_r[idx] = taubeta[self.node_index_map[cluster_receivers[idx]]]

        sender_mx = np.multiply(sen_mtx, priors)
        receiver_mx = np.multiply(rec_mtx, priors)

        sender_counts = np.divide(np.sum(sender_mx, axis=1) + tbeta_s, normalize_factor)
        receiver_counts = np.divide(np.sum(receiver_mx, axis=1) + tbeta_r, normalize_factor)

        sender_counts_log = np.log(sender_counts)
        receiver_counts_log = np.log(receiver_counts)

        lh = np.sum(sender_counts_log) + np.sum(receiver_counts_log)

        return np.around(lh, 5)

    def sample_tables_beta_dynamic(self, taubeta):
        tables = np.zeros(self.num_users)
        num_tables = 0
        uniq_clusters = np.unique(self.cluster)
        for k in uniq_clusters:
            cluster_inds = np.argwhere(self.cluster == k).flatten()
            pairs_k = self.pairs[cluster_inds]
            senders_k = pairs_k[:, 0]
            receivers_k = pairs_k[:, 1]
            sender_cluster_inds, receiver_cluster_inds = {}, {}
            for v,user in self.index_node_map.items():
                sender_cluster_inds[v] = cluster_inds[senders_k == user]
                receiver_cluster_inds[v] = cluster_inds[receivers_k == user]
                for i in range(len(sender_cluster_inds[v])): # for each time we've seen that sender:
                    if i == 0:# first time we've seen sender s, we have to start a new table
                        tables[v] += 1
                        num_tables += 1
                    else:
                        senders_before_i = sender_cluster_inds[v][:i]
                        i_ind = sender_cluster_inds[v][i]
                        prob_new_table = taubeta[v] / (np.sum(self.prior_tables[i_ind, senders_before_i]) + taubeta[v]) # we start a table with probability taubeta[s] / (taubeta[s] + sum(delay(i, senders_before_i))
                        r = np.random.rand()
                        if r < prob_new_table:
                            tables[v] += 1
                            num_tables += 1
                for i in range(len(receiver_cluster_inds[v])):
                    if i == 0:
                        tables[v] += 1
                        num_tables += 1
                    else:
                        receivers_before_i = receiver_cluster_inds[v][:i]
                        i_ind = receiver_cluster_inds[v][i]
                        prob_new_table = taubeta[v] / (np.sum(self.prior_tables[i_ind, receivers_before_i]) + taubeta[v])
                        r = np.random.rand()
                        if r < prob_new_table:
                            tables[v] += 1
                            num_tables += 1
        tables_sigma = tables - self.sigma
        self.tables_sigma = np.append(tables_sigma, self.gamma + (self.num_users * self.sigma))
        self.beta = np.random.dirichlet(self.tables_sigma)
        self.num_tables, self.tables = num_tables, tables

    def sample_alpha(self, stepsize=0.01):
        ''' 
        samples alpha, using Metropolis-Hastings. Our proposal distribution is proportional to
        p(alpha|alpha_params) p(link | alpha, f1)
        (all other terms are independent of alpha)
        '''
        alpha_prop = self.alpha + stepsize * np.random.randn() # proposal value
        
        if alpha_prop > 0: # no point proceeding if it's negative
            log_alpha_prop = np.log(alpha_prop)
            log_prior_old = ss.gamma.logpdf(self.alpha, self.alpha_params[0], scale=1./self.alpha_params[1])
            log_prior_prop = ss.gamma.logpdf(alpha_prop, self.alpha_params[0], scale=1./self.alpha_params[1])

            log_p_links_old = 0
            log_p_links_prop = 0
            log_prior_clusters_prop = self.log_prior_clusters + 0.
            for i in range(self.pairs.shape[0]):
                log_prior_clusters_prop[i, i] = log_alpha_prop
                log_p_links_old += self.log_prior_clusters[i, self.link[i]] - logsumexp(self.log_prior_clusters[i, :(i+1)])
                log_p_links_prop += log_prior_clusters_prop[i, self.link[i]] - logsumexp(log_prior_clusters_prop[i, :(i+1)])

            log_acceptance_ratio = log_p_links_prop + log_prior_prop - log_p_links_old - log_prior_old
            r = np.log(np.random.rand())
            if r < log_acceptance_ratio:
                self.alpha = alpha_prop
                self.log_prior_clusters = log_prior_clusters_prop

    def sample_gamma(self, stepsize=0.01):
        '''
        samples gamma, using Metropolis-Hastings. Our proposal distribution is proportional to
        p(gamma|gamma_params) p(beta|gamma, sigma, tables)
        (all other terms are independent of gamma)

        To get p(beta|gamma, sigma, tables), we need to sample the tables again (or, to run this just after we have sampled the tables)
        '''
        gamma_prop = self.gamma + stepsize * np.random.randn()  # proposal value

        if gamma_prop > 0:  # Assume we've just sampled tables and betas
            tables_prop = self.tables - self.sigma
            tables_prop = np.append(tables_prop, gamma_prop + (self.num_users * self.sigma))
            try:
                log_prob_prop = ss.dirichlet.logpdf(self.beta, tables_prop)
                log_prob_old = ss.dirichlet.logpdf(self.beta, self.tables_sigma)

                log_prior_old = ss.gamma.pdf(self.gamma, self.gamma_params[0], scale=1. / self.gamma_params[1])
                log_prior_prop = ss.gamma.pdf(gamma_prop, self.gamma_params[0], scale=1. / self.gamma_params[1])
            except ValueError:
                self.beta[self.beta==0] = self.epsilon
                log_prob_prop = ss.dirichlet.logpdf(self.beta, tables_prop)
                log_prob_old = ss.dirichlet.logpdf(self.beta, self.tables_sigma)

                log_prior_old = ss.gamma.pdf(self.gamma, self.gamma_params[0], scale=1. / self.gamma_params[1])
                log_prior_prop = ss.gamma.pdf(gamma_prop, self.gamma_params[0], scale=1. / self.gamma_params[1])

            log_acceptance_ratio = log_prob_prop - log_prob_old + log_prior_prop  - log_prior_old

            r = np.log(np.random.rand())

            if r < log_acceptance_ratio:
                self.gamma = gamma_prop

    def sample_tau(self,log_cluster_lhoods_old, cluster_inds, stepsize=0.01):
        '''
        samples tau, using Metropolis-Hastings. Our proposal distribution is proportional to
        p(tau|tau_params) p(data|beta, tau, f2)
        (all other terms are independent of tau)
        To calculate p(data|beta, tau, f2) we can use cluster_likelihood_dynamic -- with either taubeta calculated using self.tau, or calculated using tau_prop

        '''
        tau_prop = self.tau + stepsize * np.random.randn()  # proposal value

        if tau_prop > 0:  # Assume we've just sampled tables and betas and computed lhoods of clusters

            taubeta = tau_prop * self.beta
            log_cluster_lhoods_prop = [self.cluster_likelihood_dynamic(cluster_inds[ci], taubeta) for ci in np.unique(self.cluster)]

            log_prior_old = ss.gamma.logpdf(self.tau, self.tau_params[0], scale=1. / self.tau_params[1])
            log_prior_prop = ss.gamma.logpdf(tau_prop, self.tau_params[0], scale=1. / self.tau_params[1])
            
            log_acceptance_ratio = np.sum(log_cluster_lhoods_prop) + log_prior_prop - np.sum(log_cluster_lhoods_old) - log_prior_old

            r = np.log(np.random.rand())
            if r < log_acceptance_ratio:
                self.tau = tau_prop

    def sample_sigma(self, stepsize=0.01):
        '''
        samples sigma, using Metropolis-Hastings. Our proposal distribution is proportional to
        p(sigma|sigma_params) p(beta|gamma, sigma, tables)
        (all other terms are independent of sigma)
        Note, we must have sigma >= 0 and sigma<1, and our prior for sigma is beta, not gamma
        To get p(beta|gamma, sigma, tables), we need to sample the tables again (or, to run this just after we have sampled the tables)
        '''
        sigma_prop = self.sigma + stepsize * np.random.randn()  # proposal value

        if 0<=sigma_prop <1:  # Assume we've just sampled tables and betas
            tables_prop = self.tables - sigma_prop
            tables_prop = np.append(tables_prop, self.gamma + (self.num_users * sigma_prop))
            try:
                log_prob_prop = ss.dirichlet.logpdf(self.beta, tables_prop)
                log_prob_old = ss.dirichlet.logpdf(self.beta, self.tables_sigma)

                log_prior_old = ss.beta.logpdf(self.sigma, self.sigma_params[0], self.sigma_params[1])
                log_prior_prop = ss.beta.logpdf(sigma_prop, self.sigma_params[0], self.sigma_params[1])
            except ValueError:
                return
            log_acceptance_ratio = log_prob_prop- log_prob_old + log_prior_prop - log_prior_old

            r = np.log(np.random.rand())

            if r < log_acceptance_ratio:
                self.sigma = sigma_prop

    def sample_f1param(self, stepsize=0.01):
        '''
        samples window size of f1, using Metropolis-Hastings. Our proposal distribution is proportional to
        p(window|f1_params) p(link | window, f1)
        '''

        window_prop = self.f1_param + stepsize * np.random.uniform(-self.num_pairs, self.num_pairs,1)[0]

        if 1<window_prop <= self.num_pairs:  # no point proceeding if it's negative
            log_prior_old = ss.gamma.logpdf(self.f1_param, self.f1params[0],scale=1. / self.f1params[1])
            log_prior_prop = ss.gamma.logpdf(window_prop, self.f1params[0], scale=1. / self.f1params[1])

            log_prior_clusters_prop, x = self.get_log_priors(self.num_pairs, self.times, f1=True, f2=False,f1param=window_prop)

            log_p_links_old = 0
            log_p_links_prop = 0

            for i in range(1, self.pairs.shape[0]):
                log_p_links_old += self.log_prior_clusters[i, self.link[i]] - logsumexp(self.log_prior_clusters[i, :(i + 1)])
                log_p_links_prop += log_prior_clusters_prop[i, self.link[i]] - logsumexp(log_prior_clusters_prop[i, :(i + 1)])

            log_acceptance_ratio = log_p_links_prop + log_prior_prop - log_p_links_old - log_prior_old
            r = np.log(np.random.rand())

            if r < log_acceptance_ratio:
                self.f1_param = window_prop
                self.log_prior_clusters = log_prior_clusters_prop

    def sample_f2param(self, log_c_lhoods_old, cluster_inds, taubeta, stepsize=0.01):
        '''
        samples tau, using Metropolis-Hastings. Our proposal distribution is proportional to
        p(tau|tau_params) p(data|beta, tau, f2)
        (all other terms are independent of tau)
        To calculate p(data|beta, tau, f2) we can use cluster_likelihood_dynamic -- with either taubeta calculated using self.tau, or calculated using tau_prop

        '''
        window_prop = self.f2_param + stepsize * np.random.randn()

        if 1<window_prop <= self.num_pairs:
            x, priortables_prop = self.get_log_priors(self.num_pairs, self.times, f1=False, f2=True,f1param=None, f2param=window_prop)

            log_c_lhoods_prop = [self.cluster_likelihood_dynamic(cluster_inds[ci], taubeta, priortables_prop) for ci in
                                       np.unique(self.cluster)]

            log_prior_old = ss.gamma.logpdf(self.f2_param, self.f2params[0], scale=1. / self.f2params[1])
            log_prior_prop = ss.gamma.logpdf(window_prop, self.f2params[0], scale=1. / self.f2params[1])

            log_acceptance_ratio = np.sum(log_c_lhoods_prop) + log_prior_prop - np.sum(log_c_lhoods_old) - log_prior_old

            r = np.log(np.random.rand())
            if r < log_acceptance_ratio:
                self.f2_param = window_prop

    def infer(self, init= True,best_clusters=None):
        if init:
            if best_clusters is None:
                self.initialize_clusters_best_community(self.pairs)
                best_clusters = np.copy(self.cluster)
            else:
                self.cluster = best_clusters

        self.beta = np.ones(self.num_users + 1) / (self.num_users + 1.)
        taubeta = self.tau * self.beta

        for c in np.unique(self.cluster):
            edges_c = np.argwhere(self.cluster == c)
            self.link[edges_c[0]] = edges_c[0]
            self.link[edges_c[1:]] = edges_c[0:-1]

        # initialize log_prior_clusters, lhood and merged_lhood
        self.log_prior_clusters, self.prior_tables = self.get_log_priors(self.num_pairs, self.times)

        lhood = np.random.random(self.num_pairs)
        merged_lhood = np.random.random(self.num_pairs)
        st = time()
        for ci in np.unique(self.cluster):
            edges_inds = np.argwhere(self.cluster == ci).flatten()
            lhood[edges_inds] = self.cluster_likelihood_dynamic(edges_inds, taubeta)
        obs_lhood = []
        a_hist, g_hist, u_hist, s_hist, w1_hist, w2_hist = [],[],[],[],[],[]

        for t in range(self.n_iter):
            st= time()
            self.sample_tables_beta_dynamic(taubeta)
            taubeta = self.tau * self.beta
            log_cluster_lhoods_old=[]
            cluster_inds={}
            for ci in np.unique(self.cluster):
                edges_inds = np.argwhere(self.cluster == ci).flatten()
                log_cluster_lhood = self.cluster_likelihood_dynamic(edges_inds,taubeta)
                lhood[edges_inds] = log_cluster_lhood
                log_cluster_lhoods_old.append(log_cluster_lhood)
                cluster_inds[ci] = edges_inds

            a_hist.append(self.alpha)
            g_hist.append(self.gamma)
            u_hist.append(self.tau)
            s_hist.append(self.sigma)
            w1_hist.append(self.f1_param)
            w2_hist.append(self.f2_param)
            obs_lhood.append(np.sum([lhood[u] for u in np.unique(self.cluster)]))

            if self.sample_hypers:
                self.sample_alpha(stepsize=.1)
                self.sample_gamma(stepsize=.1)
                self.sample_tau(log_cluster_lhoods_old, cluster_inds,stepsize=.05)
                if self.f1_decay != 'crp':
                    self.sample_sigma(stepsize=.05)
                    self.sample_f1param(stepsize=0.05)
                    self.sample_f2param(log_cluster_lhoods_old, cluster_inds, taubeta, stepsize=0.1)
                taubeta = self.tau * self.beta

            for i in range(self.num_pairs):
                old_link = self.link[i]
                old_cluster = self.cluster[old_link]
                self.cluster[i] = i
                self.link[i] = i
                linked = sorted(get_linked(i, self.link))

                self.cluster[linked] = i

                if old_cluster != i:  # if we've split off to create a new self.cluster
                    edges_inds = np.argwhere(self.cluster == old_cluster).flatten()
                    lhood[old_cluster] = self.cluster_likelihood_dynamic(edges_inds, taubeta)

                lhood[i] = self.cluster_likelihood_dynamic(np.array(linked),taubeta)
                for j in np.unique(self.cluster):
                    if j == self.cluster[i]:
                        merged_lhood[j] = 2 * lhood[i]
                    elif j > i:
                        merged_lhood[j] = -np.inf
                    else:
                        edges_inds= sorted(list(set(np.append(linked,np.argwhere((self.cluster == j))))))
                        merged_lhood[j] = self.cluster_likelihood_dynamic(np.array(edges_inds), taubeta)

                log_prob = self.log_prior_clusters[i, :i+1] + merged_lhood[self.cluster[:i+1]] \
                           - lhood[self.cluster[:i+1]] - lhood[i]

                # sample new self.link and update self.cluster
                prob = np.exp(log_prob)
                prob[np.where(prob == np.inf)] = 100
                prob = prob / np.sum(prob)
                self.link[i] = np.random.multinomial(1, prob).argmax()

                new_cluster = self.cluster[self.link[i]]
                if new_cluster != i:
                    # update... if new_self.cluster == i, nothing has changed
                    self.cluster[linked] = new_cluster
                    lhood[new_cluster] = merged_lhood[new_cluster]
            if t % 100 == 0:
                print('==\t t: {}\t K: {}\t B: {}\t a:{:.2}\t g:{:.2}\t u:{:.2}\t s:{:.2}\t w1:{:.2}\t w2:{:.2}\t rt:{:.3}'
                      .format(t, len(np.unique(self.cluster)), self.num_tables, self.alpha, self.gamma, self.tau,
                              self.sigma, self.f1_param, self.f2_param, time()-st))
        return [self.cluster, self.link, obs_lhood, a_hist, g_hist, u_hist, s_hist, w1_hist, w2_hist, best_clusters]

    def testset_l2r(self, test_pairs, test_epochs, test_tstamps):
        Ntrain = self.pairs.shape[0]
        Nd = test_pairs.shape[0]
        all_tstamps = np.append(self.times, test_tstamps)
        log_cluster_priors, prior_tables = self.get_log_priors(Ntrain+Nd, all_tstamps)
        l = 0
        for n in range(Nd):
            log_p_n = np.zeros(test_epochs)
            for r in range(test_epochs):
                log_p_n[r] = self.get_log_prob_test_link(n, Ntrain, test_pairs, log_cluster_priors, prior_tables)
            p_n = logsumexp(log_p_n) - np.log(test_epochs)
            l += p_n
        return l

    def get_log_prob_test_link(self, n, Ntrain, test_pairs, log_cluster_priors, prior_tables):
        cluster = np.copy(self.cluster)
        num_users = np.max(self.pairs) + 1
        pairs = np.copy(self.pairs)
        taubeta = self.tau * self.beta
        for i in range(n+1): # also sample nth edge
            i_ind = i + Ntrain
            cluster_ids = np.unique(cluster)
            K = cluster_ids.shape[0]
            cluster_log_probs = np.zeros(K + 1)
            cluster_log_liks = np.zeros(K + 1)
            pair = np.minimum(test_pairs[i], num_users)
            s, r = pair
            for k_idx, k in enumerate(cluster_ids):
                cluster_locs = np.argwhere(cluster == k).flatten()
                cluster_log_probs[k_idx] = logsumexp(log_cluster_priors[i_ind, cluster_locs])
                cluster_norm_const = np.sum(prior_tables[i_ind, cluster_locs]) + self.tau
                sender_cluster_locs = np.argwhere((cluster == k) & (pairs[:, 0] == s)).flatten()
                receiver_cluster_locs = np.argwhere((cluster == k) & (pairs[:, 1] == r)).flatten()
                cluster_log_liks[k_idx] = np.log(
                    np.sum(prior_tables[i_ind, sender_cluster_locs]) + taubeta[self.node_index_map[s]]) - np.log(cluster_norm_const)
                cluster_log_liks[k_idx] += np.log(
                    np.sum(prior_tables[i_ind, receiver_cluster_locs]) + taubeta[self.node_index_map[r]]) - np.log(cluster_norm_const)
            cluster_log_probs[-1] = np.log(self.alpha)
            cluster_log_probs = cluster_log_probs - logsumexp(cluster_log_probs)
            cluster_log_liks[-1] = np.log(self.beta[self.node_index_map[s]]) + np.log(self.beta[self.node_index_map[r]])
            probs = cluster_log_probs + cluster_log_liks
            if i == n:
                return logsumexp(probs)
            else:
                probs -= logsumexp(probs)
                c_idx = np.random.multinomial(1, np.exp(probs)).argmax()
                if c_idx >=K:
                    cluster = np.append(cluster, i_ind)
                else:
                    cluster = np.append(cluster, cluster_ids[c_idx])
                pairs = np.append(pairs, [pair], axis=0)

        return -np.inf

    def link_prediction(self, true_pairs_next_t, num_links=100,num_sample_per_link=10):
        # assume we know number of nodes in the next timestamp to be compatible to the baselines
        num_users = np.max(self.pairs)+1
        actual = np.minimum(true_pairs_next_t[:num_links], num_users)
        num_links = min(len(actual), num_links)
        Ntrain = self.pairs.shape[0]
        self.node_index_map[num_users] = -1
        next_timestamp = self.times[-1] + 1
        all_times = np.append(self.times, next_timestamp)
        log_cluster_priors, prior_tables = self.get_log_priors(all_times.shape[0], all_times)
        test_nodes = np.unique(true_pairs_next_t[:num_links,:])
        possible_pairs = np.array([[s, r] for s in test_nodes for r in test_nodes if s<r])
        taubeta = self.beta * self.tau
        predicted = {}
        cluster_ids = np.unique(self.cluster)
        K = cluster_ids.shape[0]
        for sample in range(num_sample_per_link):
            log_probs = defaultdict(lambda:0)
            for i, pair_i in enumerate(possible_pairs):
                cluster_log_probs = np.zeros(K + 1)
                cluster_log_liks = np.zeros(K + 1)
                pair = np.minimum(pair_i, num_users)
                s, r = pair
                for k_idx, k in enumerate(cluster_ids):
                    cluster_locs = np.argwhere(self.cluster == k).flatten()
                    cluster_log_probs[k_idx] = logsumexp(log_cluster_priors[Ntrain, cluster_locs])
                    cluster_norm_const = np.sum(prior_tables[Ntrain, cluster_locs]) + self.tau
                    sender_cluster_locs = np.argwhere((self.cluster == k) & (self.pairs[:, 0] == s)).flatten()
                    receiver_cluster_locs = np.argwhere((self.cluster == k) & (self.pairs[:, 1] == r)).flatten()
                    cluster_log_liks[k_idx] = np.log(
                        np.sum(prior_tables[Ntrain, sender_cluster_locs]) + taubeta[self.node_index_map[s]]) - np.log(
                        cluster_norm_const)
                    cluster_log_liks[k_idx] += np.log(
                        np.sum(prior_tables[Ntrain, receiver_cluster_locs]) + taubeta[self.node_index_map[r]]) - np.log(
                        cluster_norm_const)
                cluster_log_probs[-1] = np.log(self.alpha)
                cluster_log_probs = cluster_log_probs - logsumexp(cluster_log_probs)
                cluster_log_liks[-1] = np.log(self.beta[self.node_index_map[s]]) + np.log(self.beta[self.node_index_map[r]])
                probs = cluster_log_probs + cluster_log_liks
                probs -= logsumexp(probs)
                c_idx = np.random.multinomial(1, np.exp(probs)).argmax()
                log_probs[(s, r)] += probs[c_idx]
            predicted[sample] = sorted(log_probs, key=log_probs.get, reverse=True)[:num_links]
            while len(predicted[sample])<num_links:
                predicted[sample].append(predicted[sample][0])

        return predicted, possible_pairs, [(s,r) for i, [s, r] in enumerate(actual)]



