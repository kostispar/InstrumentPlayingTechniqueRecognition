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
def segs2flags(seg_start, seg_end, seg_label, win_size):
    '''
    This function converts segment endpoints and respective segment
    labels to fix-sized class labels.
    ARGUMENTS:
     - seg_start:    segment start points (in seconds)
     - seg_end:    segment endpoints (in seconds)
     - seg_label:    segment labels
      - win_size:    fix-sized window (in seconds)
    RETURNS:
     - flags:    numpy array of class indices
     - class_names:    list of classnames (strings)
    '''
    flags = []
    class_names = list(set(seg_label))
    curPos = win_size / 2.0
    while curPos < seg_end[-1]:
        for i in range(len(seg_start)):
            if curPos > seg_start[i] and curPos <= seg_end[i]:
                break
        flags.append(class_names.index(seg_label[i]))
        curPos += win_size
    return numpy.array(flags), class_names



def readSegmentGT(gt_file):
    '''
    This function reads a segmentation ground truth file, following a simple CSV format with the following columns:
    <segment start>,<segment end>,<class label>

    ARGUMENTS:
     - gt_file:       the path of the CSV segment file
    RETURNS:
     - seg_start:     a numpy array of segments' start positions
     - seg_end:       a numpy array of segments' ending positions
     - seg_label:     a list of respective class labels (strings)
    '''
    f = open(gt_file, 'rt')
    reader = csv.reader(f, delimiter=',')
    seg_start = []
    seg_end = []
    seg_label = []
    for row in reader:
        if len(row) == 3:
            seg_start.append(float(row[0]))
            seg_end.append(float(row[1]))
            #if row[2]!="other":
            #    seg_label.append((row[2]))
            #else:
            #    seg_label.append("silence")
            seg_label.append((row[2]))
    return numpy.array(seg_start), numpy.array(seg_end), seg_label



def flags2segs(flags, window):
    '''
    ARGUMENTS:
     - flags:      a sequence of class flags (per time window)
     - window:     window duration (in seconds)

    RETURNS:
     - segs:       a sequence of segment's limits: segs[i,0] is start and
                   segs[i,1] are start and end point of segment i
     - classes:    a sequence of class flags: class[i] is the class ID of
                   the i-th segment
    '''

    preFlag = 0
    cur_flag = 0
    n_segs = 0

    cur_val = flags[cur_flag]
    segsList = []
    classes = []
    while (cur_flag < len(flags) - 1):
        stop = 0
        preFlag = cur_flag
        preVal = cur_val
        while (stop == 0):
            cur_flag = cur_flag + 1
            tempVal = flags[cur_flag]
            if ((tempVal != cur_val) | (cur_flag == len(flags) - 1)):  # stop
                n_segs = n_segs + 1
                stop = 1
                cur_seg = cur_val
                cur_val = flags[cur_flag]
                segsList.append((cur_flag * window))
                classes.append(preVal)
    segs = numpy.zeros((len(segsList), 2))

    for i in range(len(segsList)):
        if i > 0:
            segs[i, 0] = segsList[i-1]
        segs[i, 1] = segsList[i]
    return (segs, classes)

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
     compute_beat] = load_model(model_name)

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
    (segs, classes) = flags2segs(flags, mt_step)
    segs[-1] = len(x) / float(fs)
    if gt == True:
        # Load grount-truth:
        if os.path.isfile(gt_file):
            [seg_start_gt, seg_end_gt, seg_l_gt] = readSegmentGT(gt_file)
            flags_gt, class_names_gt = segs2flags(seg_start_gt, seg_end_gt, seg_l_gt, mt_step)
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


def load_model(model_name):
    '''
    This function loads a pre-trained model.
    It is based on pyAudioAnalysis load_model from audioTrainTest
    ARGMUMENTS:
        - model_name:     the path of the model to be loaded
    '''

    try:
        fo = open(model_name + "MEANS", "rb")
    except IOerror:
        print("Load SVM model: Didn't find file")
        return
    try:
        MEAN = cPickle.load(fo)
        STD = cPickle.load(fo)
        classNames = cPickle.load(fo)
        mt_win = cPickle.load(fo)
        mt_step = cPickle.load(fo)
        st_win = cPickle.load(fo)
        st_step = cPickle.load(fo)
        compute_beat = cPickle.load(fo)

    except:
        fo.close()
    fo.close()

    MEAN = numpy.array(MEAN)
    STD = numpy.array(STD)

    with open(model_name, 'rb') as fid:
        SVM = cPickle.load(fid)

    return (SVM, MEAN, STD, classNames, mt_win, mt_step, st_win, st_step, compute_beat)


def plotSegmentationResults(flags_ind, flags_ind_gt, class_names, mt_step, ONLY_EVALUATE=False):
    '''
    This function plots statistics on the classification-segmentation results produced either by the fix-sized supervised method or the HMM method.
    It also computes the overall accuracy achieved by the respective method if ground-truth is available.
    '''
    flags = [class_names[int(f)] for f in flags_ind]
    (segs, classes) = flags2segs(flags, mt_step)
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



