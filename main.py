import dill
import pickle

from main_classes import *
from args import Args
import time
import os
import sys
from datetime import timedelta
from util import *
from sklearn.metrics import f1_score

if __name__ == '__main__':

    args = Args(sys.argv[1:])
    if args.test_lik:
        old_stdout = sys.stdout
        log_file = open(os.path.join(os.getcwd(),
                                     args.out_path + 'log.log'), 'w')
        sys.stdout = log_file
    else:
        old_stdout = sys.stdout
        log_file = open(os.path.join(os.getcwd(),
                                     args.out_path + '{}_prediction_log.log'.format(args.begin_time)), 'w')
        sys.stdout = log_file
    for t in range(args.begin_time, args.end_time, args.duration):
        test_true_clusters, train_true_clusters = None, None,
        begin_time = time.time()
        train_pairs = np.load(args.ds_dir + 'train_pairs_{}.npy'.format(t))
        train_tstamps = np.load(args.ds_dir + 'train_tstamps_{}.npy'.format(t))//args.method_scale
        test_pairs = np.load(args.ds_dir + 'test_pairs_{}.npy'.format(t))
        test_tstamps = np.load(args.ds_dir + 'test_tstamps_{}.npy'.format(t))//args.method_scale
        num_nodes = np.max(train_pairs) + 1
        num_pairs = train_pairs.shape[0]
        print('---- time {} with {} edges {} nodes ----'.format(t, num_pairs,num_nodes))
        if not args.load_model:
            model = DynamicClusters(train_pairs,train_tstamps, args, f1params=np.array([30, 1]), f2params=np.array([30, 1]))
            cluster, link, obs_lhood, a_hist, g_hist, u_hist, s_hist\
                , w1_hist,w2_hist,besclusters = model.infer(init=False)#,fix_cluster=False)
            np.save(args.out_path + 'cluster_{}'.format(t), cluster)
            np.save(args.out_path + 'link_{}'.format(t), link)
            np.save(args.out_path + 'obs_lhood_{}'.format(t), obs_lhood)
            np.save(args.out_path + 'a_hist_{}'.format(t), a_hist)
            np.save(args.out_path + 'g_hist_{}'.format(t), g_hist)
            np.save(args.out_path + 'u_hist_{}'.format(t), u_hist)
            np.save(args.out_path + 's_hist_{}'.format(t), s_hist)
            np.save(args.out_path + 'w1_hist_{}'.format(t), w1_hist)
            np.save(args.out_path + 'w2_hist_{}'.format(t), w2_hist)

            print('time elapsed ',str(timedelta(seconds = time.time()-begin_time)))
            with open(args.out_path + 'model_{}.obj'.format(t, args.alpha, args.gamma, args.tau, args.sigma),'wb') as f:
                dill.dump(model, f)
            args.alpha, args.gamma, args.tau, args.sigma, args.f1_param, args.f2_param = \
                model.alpha, model.gamma, model.tau, model.sigma, model.f1_param, model.f2_param
        else:
            with open(
                    args.out_path + 'model_{}.obj'.format(t, args.alpha, args.gamma, args.tau, args.sigma),
                    'rb') as f:
                model = dill.load(f)
        if test_pairs.shape[0]>0 and args.test_lik:
            v_test = np.unique(test_pairs).shape[0]
            test_set_loglik = model.testset_l2r(test_pairs, args.num_samples, test_tstamps)#, test_true_clusters, train_true_clusters)
            num_users  = np.unique(train_pairs).shape[0]
            print('sanity ', test_pairs.shape[0] * (np.log(1. / (num_users * num_users))))
            print('test_set_loglik ',test_set_loglik)
            if not os.path.isfile(args.out_path + 'test_log_lik.txt'):
                with open(args.out_path + 'test_log_lik.txt', 'w'): pass
            with open(args.out_path + 'test_log_lik.txt', 'a') as f:
                f.write('{};{};{};{};{}\n'.format(t, train_pairs.shape[0], test_pairs.shape[0],
                                                     str(timedelta(seconds=time.time() - begin_time)),
                                                     test_set_loglik))

        if not args.test_lik:
            pairs_next_t = np.load(args.ds_dir + '/t_pairs_{}.npy'.format(t + 1))
            predicted, all_elements, actual = model.link_prediction(pairs_next_t, args.num_links, args.num_samples)
            with open(args.out_path + 'link_predicted_{}.pkl'.format(t), 'wb') as f:
                pickle.dump(predicted, f)
            f1_scores = np.zeros(args.num_samples)
            scores = {}
            hits = np.zeros((args.num_samples, len(args.k_list)))
            aps = np.zeros((args.num_samples, len(args.k_list)))
            y_ = get_binary_mtx(actual, all_elements)
            for sample in range(args.num_samples):
                y_pred = get_binary_mtx(predicted[sample], all_elements)
                f1_scores[sample] = f1_score(y_, y_pred)
                scores[sample], hits[sample], aps[sample] = portfolio(predicted[sample], actual, k_list=args.k_list)
            # print('****F1****',np.mean(f1_scores),np.std(f1_scores))
            # print('****scores***',scores)
            # print('**** Hits ****',np.mean(hits, axis=0),np.std(hits, axis=0))
            # print('**** APs ****',np.mean(aps, axis=0),np.std(aps, axis=0))
            if not os.path.isfile(args.out_path + 'prediction_evals_{}.txt'.format(args.begin_time)):
                with open(args.out_path + 'prediction_evals_{}.txt'.format(args.begin_time), 'w'): pass
            with open(args.out_path + 'prediction_evals_{}.txt'.format(args.begin_time), 'a') as f:
                f.write('{};{};{};{};{};{};{}\n'.format(t, np.mean(f1_scores),np.std(f1_scores), np.mean(hits, axis=0)
                                                        ,np.std(hits, axis=0), np.mean(aps, axis=0),np.std(aps, axis=0)))

    sys.stdout = old_stdout
    log_file.close()
