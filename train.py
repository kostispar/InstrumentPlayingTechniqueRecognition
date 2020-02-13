from __future__ import print_function
from pyAudioAnalysis import audioTrainTest as aT
import argparse

def parseArguments():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-d' , '--data_folders', nargs="+")
    parser.add_argument('-c', '--classifier_type', nargs=None, required=True,
                        choices = ["knn", "svm", "svm_rbf", "randomforest",
                                   "extratrees", "gradientboosting"],
                        help="Classifier type")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parseArguments()
    data_paths = args.data_folders
    classifier_type = args.classifier_type

    mt_win, mt_step, st_win, st_step = 0.9, 0.6, 0.06, 0.02
    aT.extract_features_and_train(data_paths,
                                  mt_win, mt_step, st_win, st_step,
                                  classifier_type,
                                  "model_" + classifier_type, False)