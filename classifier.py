import pickle as cPickle
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioSegmentation as aS
import numpy
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import re


def FileClassification(input_file, model_name, model_type, gt = False, gt_file=""):
    '''
    This function opens the input file with audioBasicIO (pyAudioAnalysis) and loads the pre-trained models.
    Using audioFeatureExtraction (pyAudioAnalysis), the audio features are extracted and the pre-trained classifiers classify
    the segments of the audio stream. The segment classifications are plotted.
    FileClassification has the same functionality with pyAudioAnalysis mtFileClassification:

    ARGUMENTS:
        - input_file:       path of the input WAV file
        - model_name:       name of the classification model
        - model_type:       svm or knn depending on the classifier type
        - confidence:       the Probability confidence, above which the classifier's predictions are accepted.
                            In other case, the 'None technique' is assigned to the segment
    '''

    if not os.path.isfile(model_name):
        print("mtFileClassificationError: input model_type not found!")
        return (-1, -1, -1, -1)

    # Load classifier with load_model:
    [classifier, MEAN, STD, class_names, mt_win, mt_step, st_win, st_step,
     compute_beat] = aT.load_model(model_name)

    # Using audioBasicIO from puAudioAnalysis, the input audio stream is loaded
    [fs, x] = audioBasicIO.readAudioFile(input_file)
    if fs == -1:  # could not read file
        return (-1, -1, -1, -1)
    x = audioBasicIO.stereo2mono(x)  # convert stereo (if) to mono
    duration = len(x) / fs

    # mid-term feature extraction using pyAudioAnalysis mtFeatureExtraction:
    [mt_feats, _, _] = aF.mtFeatureExtraction(x, fs, mt_win * fs,
                                              mt_step * fs,
                                              round(fs * st_win),
                                              round(fs * st_step))

    flags = []
    Ps = []
    flags_ind = []
    for i in range(mt_feats.shape[1]):  # for each feature vector (i.e. for each fix-sized segment):
        cur_fv = (mt_feats[:, i] - MEAN) / STD  # normalize current feature vector
        [res, P] = aT.classifierWrapper(classifier, model_type, cur_fv)  # classify vector
        if res == 0.0:
            if numpy.max(P) > 0.3:
                flags_ind.append(res)
                flags.append(class_names[int(res)])  # update class label matrix
                Ps.append(numpy.max(P))  # update probability matrix
            else:
                flags_ind.append(-1)
                flags.append('None')
                Ps.append(-1)
        if res == 1.0:
            if numpy.max(P) > 0.3:
                flags_ind.append(res)
                flags.append(class_names[int(res)])  # update class label matrix
                Ps.append(numpy.max(P))  # update probability matrix
            else:
                flags_ind.append(-1)
                flags.append('None')
                Ps.append(-1)
        if res == 2.0:
            if numpy.max(P) > 0.6:
                flags_ind.append(res)
                flags.append(class_names[int(res)])  # update class label matrix
                Ps.append(numpy.max(P))  # update probability matrix
            else:
                flags_ind.append(-1)
                flags.append('None')
                Ps.append(-1)
        if res == 3.0:
            if numpy.max(P) > 0.5:
                flags_ind.append(res)
                flags.append(class_names[int(res)])  # update class label matrix
                Ps.append(numpy.max(P))  # update probability matrix
            else:
                flags_ind.append(-1)
                flags.append('None')
                Ps.append(-1)
        if res == 4.0:
            if numpy.max(P) > 0.9:
                flags_ind.append(res)
                flags.append(class_names[int(res)])  # update class label matrix
                Ps.append(numpy.max(P))  # update probability matrix
            else:
                flags_ind.append(-1)
                flags.append('None')
                Ps.append(-1)
    flags_ind = numpy.array(flags_ind)

    # 1-window smoothing
    for i in range(1, len(flags_ind) - 1):
        if flags_ind[i - 1] == flags_ind[i + 1]:
            flags_ind[i] = flags_ind[i + 1]
    # convert fix-sized flags to segments and classes
    (segs, classes) = aS.flags2segs(flags, mt_step)
    segs[-1] = len(x) / float(fs)
    if gt == True:
        # Load grount-truth:
        if os.path.isfile(gt_file):
            [seg_start_gt, seg_end_gt, seg_l_gt] = aS.readSegmentGT(gt_file)
            flags_gt, class_names_gt = aS.segs2flags(seg_start_gt, seg_end_gt, seg_l_gt, mt_step)
            flags_ind_gt = []
            # print(class_names)
            for j, fl in enumerate(flags_gt):
                # "align" labels with GT
                # print(class_names_gt[flags_gt[j]])
                if class_names_gt[flags_gt[j]] in class_names:
                    flags_ind_gt.append(class_names.index(class_names_gt[flags_gt[j]]))
                else:
                    flags_ind_gt.append(-1)
            flags_ind_gt = numpy.array(flags_ind_gt)
            cm = numpy.zeros((len(class_names_gt), len(class_names_gt)))
            for i in range(min(flags_ind.shape[0], flags_ind_gt.shape[0])):
                cm[int(flags_ind_gt[i]), int(flags_ind[i])] += 1
        else:
            cm = []
            flags_ind_gt = numpy.array([])
        acc = plotSegmentationResults(flags_ind, flags_ind_gt,
                                      class_names, mt_step, False)
    else:
        cm = []
        flags_ind_gt = numpy.array([])
        acc = plotSegmentationResults(flags_ind, flags_ind_gt,
                                      class_names, mt_step, False)
    if acc >= 0:
        print("Overall Accuracy: {0:.3f}".format(acc))
        return (flags_ind, class_names_gt, acc, cm)
    else:
        return (flags_ind, class_names, acc, cm)



def plotSegmentationResults(flags_ind, flags_ind_gt, class_names, mt_step, ONLY_EVALUATE=False):
    '''
    This function plots statistics on the classification-segmentation results produced either by the fix-sized supervised method or the HMM method.
    It also computes the overall accuracy achieved by the respective method if ground-truth is available.
    '''
    flags = [class_names[int(f)] for f in flags_ind]
    (segs, classes) = aS.flags2segs(flags, mt_step)
    min_len = min(flags_ind.shape[0], flags_ind_gt.shape[0])
    if min_len > 0:
        accuracy = numpy.sum(flags_ind[0:min_len] ==
                             flags_ind_gt[0:min_len]) / float(min_len)
    else:
        accuracy = -1

    if not ONLY_EVALUATE:
        duration = segs[-1, 1]
        s_percentages = numpy.zeros((len(class_names), 1))
        percentages = numpy.zeros((len(class_names), 1))
        av_durations = numpy.zeros((len(class_names), 1))

        for iSeg in range(segs.shape[0]):
            s_percentages[class_names.index(classes[iSeg])] += \
                (segs[iSeg, 1]-segs[iSeg, 0])

        for i in range(s_percentages.shape[0]):
            percentages[i] = 100.0 * s_percentages[i] / duration
            S = sum(1 for c in classes if c == class_names[i])
            if S > 0:
                av_durations[i] = s_percentages[i] / S
            else:
                av_durations[i] = 0.0
        class_names = ['trembling1', 'trembling2','slurring','2strings','chord']
        font = {'size': 10}
        plt.rc('font', **font)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_yticks(numpy.array(range(len(class_names))))
        ax1.axis((0, duration, -1, len(class_names)))
        ax1.set_yticklabels(class_names)
        ax1.plot(numpy.array(range(len(flags_ind))) * mt_step +
                 mt_step / 2.0, flags_ind)
        if flags_ind_gt.shape[0] > 0:
            ax1.plot(numpy.array(range(len(flags_ind_gt))) * mt_step +
                     mt_step / 2.0, flags_ind_gt + 0.05, '*r')
        plt.xlabel("time (seconds)")
        if accuracy >= 0:
            plt.title('Accuracy = {0:.1f}%'.format(100.0 * accuracy))

        ax2 = fig.add_subplot(223)
        plt.title("Classes percentage durations")
        ax2.axis((0, len(class_names) + 1, 0, 100))
        ax2.set_xticks(numpy.array(range(len(class_names) + 1)))
        ax2.set_xticklabels([" "] + class_names)
        ax2.bar(numpy.array(range(len(class_names))) + 0.5, percentages)

        ax3 = fig.add_subplot(224)
        plt.title("Segment average duration per class")
        ax3.axis((0, len(class_names)+1, 0, av_durations.max()))
        ax3.set_xticks(numpy.array(range(len(class_names) + 1)))
        ax3.set_xticklabels([" "] + class_names)
        ax3.bar(numpy.array(range(len(class_names))) + 0.5, av_durations)
        fig.tight_layout()
        plt.show()
        return accuracy


def main():
    modelpath = sys.argv[1]  # the path where the models are stored
    filepath = sys.argv[2]  # the path to the recording
    model_name = 'svm'
    if len(sys.argv) == 4:
        ground_truth_path = sys.argv[3]
        if os.path.isfile(ground_truth_path):
            gt = True
            FileClassification(filepath, modelpath, model_name, gt, ground_truth_path)
    else:
        gt = False
        FileClassification(filepath, modelpath, model_name, gt, '')


if __name__ == "__main__":
    main()



