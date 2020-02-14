from pyAudioAnalysis import MidTermFeatures as mF
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioSegmentation as aS
import numpy
import os
import sys


def FileClassification(input_file, model_name, model_type, gt=False,
                       gt_file=""):
    '''
    TODO: This function needs to be refactored according to the code in
    audioSegmentation.mid_term_file_classification()
    '''

    if not os.path.isfile(model_name):
        print("mtFileClassificationError: input model_type not found!")
        return (-1, -1, -1, -1)

    # Load classifier with load_model:
    [classifier, MEAN, STD, class_names, mt_win, mt_step, st_win, st_step,
     compute_beat] = aT.load_model(model_name)

    # Using audioBasicIO from puAudioAnalysis, the input audio stream is loaded
    [fs, x] = audioBasicIO.read_audio_file(input_file)
    if fs == -1:  # could not read file
        return (-1, -1, -1, -1)
    x = audioBasicIO.stereo_to_mono(x)  # convert stereo (if) to mono
    duration = len(x) / fs

    # mid-term feature extraction using pyAudioAnalysis mtFeatureExtraction:
    [mt_feats, _, _] = mF.mid_feature_extraction(x, fs, mt_win * fs,
                                                 mt_step * fs,
                                                 round(fs * st_win),
                                                 round(fs * st_step))
    flags = []
    Ps = []
    flags_ind = []
    for i in range(mt_feats.shape[1]):
        # for each feature vector (i.e. for each fix-sized segment):
        cur_fv = (mt_feats[:, i] - MEAN) / STD
        [res, P] = aT.classifier_wrapper(classifier, model_type, cur_fv)
        if res == 0.0:
            if numpy.max(P) > 0.5:
                flags_ind.append(res)
                flags.append(class_names[int(res)])  # update class label matrix
                Ps.append(numpy.max(P))  # update probability matrix
            else:
                flags_ind.append(-1)
                flags.append('None')
                Ps.append(-1)
        if res == 1.0:
            if numpy.max(P) > 0.9:
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
            if numpy.max(P) > 0.3:
                flags_ind.append(res)
                flags.append(class_names[int(res)])  # update class label matrix
                Ps.append(numpy.max(P))  # update probability matrix
            else:
                flags_ind.append(-1)
                flags.append('None')
                Ps.append(-1)
        if res == 4.0:
            if numpy.max(P) > 0.3:
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
    (segs, classes) = aS.labels_to_segments(flags, mt_step)
    segs[-1] = len(x) / float(fs)
    if gt == True:
        # Load grount-truth:
        if os.path.isfile(gt_file):
            [seg_start_gt, seg_end_gt, seg_l_gt] = aS.read_segmentation_gt(gt_file)
            flags_gt, class_names_gt = aS.segments_to_labels(seg_start_gt, seg_end_gt, seg_l_gt, mt_step)
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
        acc = aS.plot_segmentation_results(flags_ind, flags_ind_gt,
                                           class_names, mt_step, False)
    else:
        cm = []
        flags_ind_gt = numpy.array([])
        acc = aS.plot_segmentation_results(flags_ind, flags_ind_gt,
                                           class_names, mt_step, False)
    if acc >= 0:
        print("Overall Accuracy: {0:.3f}".format(acc))
        return (flags_ind, class_names_gt, acc, cm)
    else:
        return (flags_ind, class_names, acc, cm)


def main():
    modelpath = sys.argv[1]  # the path where the models are stored
    filepath = sys.argv[2]   # the path to the recording
    model_name = 'svm_rbf'
    if len(sys.argv) == 4:
        ground_truth_path = sys.argv[3]
        if os.path.isfile(ground_truth_path):
            gt = True
            FileClassification(filepath, modelpath, model_name, gt,
                               ground_truth_path)
    else:
        gt = False
        FileClassification(filepath, modelpath, model_name, gt, '')


if __name__ == "__main__":
    main()



