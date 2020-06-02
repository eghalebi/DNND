from os import path, makedirs


class Args():

    def __init__(self, sysargs):

        self.begin_time, self.end_time, self.duration, self.alpha, self.gamma, self.tau, self.sigma, self.out_dir, self.ds_dir \
            , self.epochs, self.num_samples, self.test_ratio, self.f1, self.f1_param, self.f2, self.f2_param, self.method_scale \
            , self.sample_hypers, self.load_model, self.test_lik, self.cpu_num, self.num_links = sysargs

        self.alpha = float(self.alpha)
        self.gamma = float(self.gamma)
        self.tau = float(self.tau)
        self.sigma = float(self.sigma)

        self.out_dir = str(self.out_dir)
        self.ds_dir = str(self.ds_dir)
        self.begin_time, self.end_time, self.duration = int(self.begin_time), int(self.end_time), int(self.duration)

        self.epochs = int(self.epochs)
        self.num_samples = int(self.num_samples)
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

        if not path.isdir(self.out_dir):
            makedirs(self.out_dir)
        if not path.isdir(self.ds_dir):
            makedirs(self.ds_dir)

        self.out_path = self.out_dir + self.f1 + '_'

        self.k_list = [10, 20, 50]
