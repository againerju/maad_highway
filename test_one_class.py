import os
import argparse
import yaml
import json
import joblib
import shutil
import pickle

import src.utils
import src.evaluation

if __name__ == "__main__":

    ### CONFIG ###

    # config file
    parser = argparse.ArgumentParser(description="Test One-class (KDE or OC-SVM) model.")
    parser.add_argument('--config', type=str, default="config_one-class_test.yaml")

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

    # feature path
    feature_path_test = os.path.join(exp_root, config["feature_encoder_directory"],
                                     config["latent_feature_directory"], "prediction_data.pkl")

    # experiment path
    run_root = os.path.join(exp_root, config["run_name"])

    # model path
    model_path = os.path.join(run_root, "one_class_model.joblib")

    # create evaluation directory
    eval_dir = "eval_" + src.utils.get_current_time()
    eval_path = os.path.join(run_root, eval_dir)

    if not os.path.isdir(eval_path):
        os.makedirs(eval_path)

    ### DATA
    data_test = src.utils.load_prediction_data(feature_path_test)
    X, segment_shapes = src.utils.get_latent_features(data_test)

    ### LOG
    copy_config_dst = os.path.join(eval_path, "config_test.yaml")
    shutil.copy(config_path, copy_config_dst)

    ### MODEL
    model = joblib.load(model_path)

    ### PREDICTION
    anomaly_scores = -model.score_samples(X)

    # add anomaly score to data dictionary
    prediction_data = src.utils.add_scores_to_data_dict(data_test, segment_shapes, anomaly_scores)

    ### EVALUATION

    # create evaluation dictionary
    eval_dict = src.evaluation.get_eval_dict_from(prediction_data, model_type="one-class")

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
