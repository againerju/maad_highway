import os
import argparse
import yaml
import joblib
import time
import datetime
import shutil
from sklearn import svm
from sklearn.neighbors import KernelDensity

import src.utils


if __name__ == "__main__":

    ### CONFIG ###

    # config file
    parser = argparse.ArgumentParser(description="Train One-class (KDE or OC-SVM) model.")
    parser.add_argument('--config', type=str, default="config_one-class_train.yaml")

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
    feature_path_train = os.path.join(exp_root, config["feature_encoder_directory"],
                                      config["latent_feature_directory"], "prediction_data.pkl")

    # experiment path
    method = config["method"]
    run_name = method
    date_time = src.utils.get_current_time() + "_"
    run_name = date_time + run_name
    exp_dir = os.path.join(exp_root, run_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    ### DATA
    data_train = src.utils.load_prediction_data(feature_path_train)
    X, _ = src.utils.get_latent_features(data_train)

    ### LOG
    copy_config_dst = os.path.join(exp_dir, "config_train.yaml")
    shutil.copy(config_path, copy_config_dst)

    ### TRAIN
    if method == 'ocsvm':

        print('Training {}...'.format(method))

        # parameterize OC-SVM
        model = svm.OneClassSVM(nu=config["ocsvm_nu"], kernel='rbf', gamma=config["ocsvm_gamma"])

        # fit
        start = time.time()
        model.fit(X)
        end = time.time()

    elif method == 'kde':
        print('Training {}...'.format(method))

        # parameterize KDe
        model = KernelDensity(kernel='gaussian', bandwidth=float(config["kde_bandwidth"]))

        # fit
        start = time.time()
        model.fit(X)
        end = time.time()

    ### EXPORT & FINISH

    # summarize training
    print("\nModel {}\nTraining time: {}".format(method.upper(), str(datetime.timedelta(seconds=end-start))))

    # export model
    model_name = method
    joblib.dump(model, os.path.join(exp_dir, "one_class_model.joblib"))

    # end training
    print("Training of {} done.".format(run_name))
