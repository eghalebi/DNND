from os import path, makedirs


class Args():

    def __init__(self, sysargs):
        '''
        begin_time: the first time slot
        end_time: the last time slot
        duration: lenght of time slot
        alpha: initial value of alpha, governs the number of clusters
        gamma: initial value of gamma, governs the overall number of vertices
        tau: initial value of tau, governs the similarity between clusters
        sigma: initial value of sigma, controls the sparsity of the multigraph
        ds_dir: path to dataset
        epochs: number of samples (iterations of MCMC), default: 1000
        f1: defines the type of f1 function, options are: crp, window, logistic, exponential
        f2: defines the type of f2 function, options are: crp, window, logistic, exponential
        f1_param: window length of f1
        f2_param: window length of f2
        num_runs: number of runs for prediction tasks, default: 10
        test_ratio: percentage of test set data, default: 0.15
        method_scale: dependong on time slots, this value scales window length
        sample_hypers: 1 if want to train on hyper-parameters, default: 1
        load_model: 1 if upload a saved model, default: 0
        test_lik: 1 for held-out edges prediction, default: 1
        cpu_num: number of CPU cores available for parallel setting, default: 1
        num_links: number of future interaction of next time slot to predict, default: 100
        
        '''

        self.begin_time, self.end_time, self.duration, self.alpha, self.gamma, self.tau, self.sigma, self.ds_dir \
            , self.epochs, self.num_runs, self.test_ratio, self.f1, self.f1_param, self.f2, self.f2_param, self.method_scale \
            , self.sample_hypers, self.load_model, self.test_lik, self.cpu_num, self.num_links = sysargs

        self.alpha = float(self.alpha)
        self.gamma = float(self.gamma)
        self.tau = float(self.tau)
        self.sigma = float(self.sigma)

        self.out_dir = str(self.out_dir)
        self.ds_dir = str(self.ds_dir)
        self.begin_time, self.end_time, self.duration = int(self.begin_time), int(self.end_time), int(self.duration)

        self.epochs = int(self.epochs)
        self.num_runs = int(self.num_runs)
        self.test_ratio = float(self.test_ratio)
        self.f1 = str(self.f1)
        self.f1_param = float(self.f1_param)
        self.f2 = str(self.f2)
        self.f2_param = float(self.f2_param)
        self.method_scale = float(self.method_scale)

        self.sample_hypers = int(self.sample_hypers)
        self.load_model = int(self.load_model)
        self.test_lik = int(self.test_lik)
        self.cpu_num = int(self.cpu_num)
        self.num_links = int(self.num_links)

        if not path.isdir(self.ds_dir):
            makedirs(self.ds_dir)

        self.k_list = [10, 20, 50] # for AP@k and hits@k evaluations
