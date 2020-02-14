from __future__ import print_function
from pyAudioAnalysis import audioTrainTest as aT
import argparse

def parseArguments():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-d' , '--data_folders', nargs="+")
    parser.add_argument('-c', '--classifier_type', nargs=None, required=True,
                        choices=["knn", "svm", "svm_rbf", "randomforest",
                                   "extratrees", "gradientboosting"],
                        help="Classifier type")
    parser.add_argument("-ws", "--segment_window", type=float,
                        choices=[0.2, 0.5, 0.8, 1.0, 1.5, 2.0],
                        default=1.0, help="Segment size (seconds)")
    parser.add_argument("-wf", "--frame_window", type=float,
                        choices=[0.020, 0.040, 0.050, 0.1],
                        default=0.050, help="Short-frame segment length "
                                            "(seconds)")

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parseArguments()
    data_paths = args.data_folders
    classifier_type = args.classifier_type

    mt_win = args.segment_window
    mt_step = mt_win / 2.0
    st_win = args.frame_window
    st_step = st_win / 2.0

    model_name = "{0:s}_{1:s}_{2:.3f}_{3:.3f}.model".format("model",
                                                            classifier_type,
                                                            mt_win,
                                                            st_win)

    aT.extract_features_and_train(data_paths,
                                  mt_win, mt_step, st_win, st_step,
                                  classifier_type,
                                  model_name, False)