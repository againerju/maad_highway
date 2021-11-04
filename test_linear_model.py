import os
import sys
import yaml
import json
import time
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import src.utils
import src.dataset
import src.evaluation

if __name__ == "__main__":

    # config file
    parser = argparse.ArgumentParser(description="Test linear model.")
    parser.add_argument('--config', type=str, default="config_linear_test.yaml")

    args = parser.parse_args()

    ### END CONFIG ###

    ### PATHS & CONFIG
    project_root = os.getcwd()
    data_root = os.path.join(project_root, "datasets/maad")
    exp_root = os.path.join(project_root, "experiments")
    config_root = os.path.join(project_root, "config")

    # config
    config_path = os.path.join(config_root, args.config)
    with open(config_path, "r") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)

    # data
    data_path_test = os.path.join(data_root, config["dataset"]["set"])

    # experiment path
    method = config["model"]["type"]
    run_name = method
    date_time = src.utils.get_current_time()
    run_name = date_time + "_" + run_name
    exp_dir = os.path.join(exp_root, run_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # create evaluation directory
    eval_dir = "eval_" + src.utils.get_current_time()
    eval_path = os.path.join(exp_dir, eval_dir)

    if not os.path.isdir(eval_path):
        os.makedirs(eval_path)

    ### DATA
    dset_test = src.dataset.MAADDataset(data_path_test, obs_len=config["model"]["obs_len"],
                                        adj_type="identity")

    loader_test = DataLoader(dset_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1)

    ### PREDICTION
    print("\nPredicting...")
    pred_start = time.time()
    prediction_data = {}
    step = 0

    for cnt, batch in enumerate(loader_test):

        step += 1

        # get data
        obs_traj, obs_traj_rel, frame_ids, seq_ids, labels, V_obs, A_obs = batch

        # prepare data
        obs_traj = obs_traj.numpy()[0]  # its anyway batch size = 1
        frame_ids = frame_ids.numpy()[0].tolist()
        seq_ids = seq_ids.numpy()[0]
        labels = labels.numpy()[0]

        # init linear trajectory
        linear_traj = np.zeros(obs_traj.shape)

        N = obs_traj.shape[0]

        # model each agent individually
        for i in range(N):

            # get agent trajectory
            agent_traj = obs_traj[i]

            # trajectory features
            start_pos = agent_traj[:, 0]
            end_pos = agent_traj[:, -1]
            n_ts = agent_traj.shape[1]

            if method == "cvm":

                # CVM
                velocity = agent_traj[:, 1] - agent_traj[:, 0]
                approx_agent_traj = np.zeros(agent_traj.shape) + velocity[:, np.newaxis]
                approx_agent_traj[:, 0] = start_pos
                approx_agent_traj = np.cumsum(approx_agent_traj, axis=1)

            elif method == "lti":

                # LTI
                x_interp = np.linspace(start_pos[0], end_pos[0], n_ts)
                y_interp = np.linspace(start_pos[1], end_pos[1], n_ts)
                approx_agent_traj = np.zeros(agent_traj.shape)
                approx_agent_traj[0] = x_interp
                approx_agent_traj[1] = y_interp

            else:

                sys.exit("Unknown model type {}. Abort!".format(method))

            # add to matrix
            linear_traj[i] = approx_agent_traj

            if True:
                plt.plot(agent_traj[0], agent_traj[1])
                plt.plot(approx_agent_traj[0], approx_agent_traj[1])

        # prepare data for dict export
        obs_traj = np.transpose(obs_traj, (2, 0, 1))
        linear_traj = np.transpose(linear_traj, (2, 0, 1))
        frame_ids = np.transpose(frame_ids, (2, 0, 1))
        labels = np.transpose(labels, (2, 0, 1))

        # frame_ids [TxNx1], seq_id [Nx1], labels [TxNx2], traj [TxNxF]
        prediction_data[step] = dict()
        prediction_data[step]['fid'] = frame_ids
        prediction_data[step]['seq_id'] = seq_ids
        prediction_data[step]['labels'] = labels
        prediction_data[step]['obs'] = obs_traj
        prediction_data[step]['trgt'] = obs_traj
        prediction_data[step]['pred'] = [linear_traj]

    pred_end = time.time()

    ### EVALUATION
    print("Evaluating...")

    # create evaluation dictionary
    eval_dict = src.evaluation.get_eval_dict_from(prediction_data, model_type="reconstruction")

    # y_true and y_pred from eval dict
    y_true, y_score = src.evaluation.get_y_true_y_pred(eval_dict)

    # compute metrics only iff more than one classes occurs in y_true
    if not src.utils.all_equal(y_true):

        # filter ignore regions
        y_true, y_score = src.evaluation.filter_ignore_regions(y_true, y_score)

        # ROC
        fpr, tpr, thresholds = src.evaluation.ad_roc(y_true, y_score)

        # AUROC
        auroc = src.evaluation.ad_auroc(y_true, y_score)

        # AUPR-abnormal
        ap_abnormal = src.evaluation.ad_aupr(y_true, y_score)

        # AUPR-normal
        ap_normal = src.evaluation.ad_aupr(1 - y_true, -y_score)

        # FPR @ 95-TPR
        fpr_at_95tpr = src.evaluation.fpr_at_95tpr(tpr, fpr)

        # log results
        results = {"auroc": auroc, "aupr-abnormal": ap_abnormal, "aupr-normal": ap_normal, "fpr @ 95-tpr": fpr_at_95tpr}
        for mname, mval in results.items():
            print("{:20}: {:8.6f}".format(mname, mval))

        # export results
        output_file = open(os.path.join(eval_path, "results.json"), "w")
        json.dump(results, output_file)

    ### EXPORT
    # export raw prediction data
    output_file = open(os.path.join(eval_path, "prediction_data.pkl"), "wb")
    pickle.dump(prediction_data, output_file)
    output_file.close()

    # export eval data
    output_file = open(os.path.join(eval_path, "eval_data.pkl"), "wb")
    pickle.dump(eval_dict, output_file)
    output_file.close()

    print("Done Testing.")
