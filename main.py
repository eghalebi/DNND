import dill
import pickle
from main_classes import *
from args import Args
import os
import sys
from util import *

if __name__ == '__main__':

    # read arguments from input
    args = Args(sys.argv[1:])
    
    # save logs 
    old_stdout = sys.stdout
    log_file = open(os.path.join(os.getcwd(), args.out_path + 'log.log'), 'w')
    sys.stdout = log_file

    for t in range(args.begin_time, args.end_time, args.duration):
        test_true_clusters, train_true_clusters = None, None # for comparison purposes
        train_pairs = np.load(args.ds_dir + 'train_pairs_{}.npy'.format(t))
        train_tstamps = np.load(args.ds_dir + 'train_tstamps_{}.npy'.format(t))//args.method_scale
        test_pairs = np.load(args.ds_dir + 'test_pairs_{}.npy'.format(t))
        test_tstamps = np.load(args.ds_dir + 'test_tstamps_{}.npy'.format(t))//args.method_scale
        print('---- time {} with {} edges {} nodes ----'.format(t, train_pairs.shape[0],np.max(train_pairs) + 1))
        if not args.load_model:
            model = DynamicClusters(train_pairs, train_tstamps, args, f1params=np.array([30, 1]), f2params=np.array([30, 1]))
            cluster, link, obs_lhood, a_hist, g_hist, u_hist, s_hist, w1_hist,w2_hist,besclusters = model.infer(init=False)
            with open(args.out_path + 'model_{}.obj'.format(t, args.alpha, args.gamma, args.tau, args.sigma),'wb') as f:
                dill.dump(model, f)
            args.alpha, args.gamma, args.tau, args.sigma, args.f1_param, args.f2_param = model.alpha, model.gamma, model.tau, model.sigma, model.f1_param, model.f2_param
        else:
            with open(args.out_path + 'model_{}.obj'.format(t, args.alpha, args.gamma, args.tau, args.sigma),'rb') as f:
                model = dill.load(f)
        if test_pairs.shape[0]>0 and args.test_lik:
            test_set_loglik = model.testset_l2r(test_pairs, args.num_samples, test_tstamps)
        if not args.test_lik:
            pairs_next_t = np.load(args.ds_dir + '/t_pairs_{}.npy'.format(t + 1))
            predicted, all_elements, actual = model.link_prediction(pairs_next_t, args.num_links, args.num_samples)
            with open(args.out_path + 'link_predicted_{}.pkl'.format(t), 'wb') as f:
                pickle.dump(predicted, f)
         
    sys.stdout = old_stdout
    log_file.close()
