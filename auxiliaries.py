import numpy as np, time, random, csv, glob
import torch, ast, pandas as pd, copy, itertools as it, os, torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
import itertools as it, copy
import matplotlib.pyplot as plt
from sklearn import metrics


### Function to extract setup info from text file ###
def extract_setup_info(config_file):
    baseline_setup = pd.read_csv(config_file, sep='\t', header=None)
    baseline_setup = [x for x in baseline_setup[0] if '=' not in x]
    sub_setups = [x.split('#')[-1] for x in np.array(baseline_setup) if '#' in x]
    vals = [x for x in np.array(baseline_setup)]
    set_idxs = [i for i, x in enumerate(np.array(baseline_setup)) if '#' in x] + [len(vals)]

    settings = {}
    for i in range(len(set_idxs) - 1):
        settings[sub_setups[i]] = [[y.replace(" ", "") for y in x.split(':')] for x in
                                   vals[set_idxs[i] + 1:set_idxs[i + 1]]]

    # d_opt = vars(opt)
    d_opt = {}
    for key in settings.keys():
        d_opt[key] = {subkey: ast.literal_eval(x) for subkey, x in settings[key]}

    opt = argparse.Namespace(**d_opt)
    if d_opt['Paths']['network_variation_setup_file'] == '':
        return [opt]

    variation_setup = pd.read_csv(d_opt['Paths']['network_variation_setup_file'], sep='\t', header=None)
    variation_setup = [x for x in variation_setup[0] if '=' not in x]
    sub_setups = [x.split('#')[-1] for x in np.array(variation_setup) if '#' in x]
    vals = [x for x in np.array(variation_setup)]
    set_idxs = [i for i, x in enumerate(np.array(variation_setup)) if '#' in x] + [len(vals)]
    settings = {}
    for i in range(len(set_idxs) - 1):
        settings[sub_setups[i]] = []
        for x in vals[set_idxs[i] + 1:set_idxs[i + 1]]:
            y = x.split(':')
            settings[sub_setups[i]].append([[y[0].replace(" ", "")], ast.literal_eval(y[1].replace(" ", ""))])
        # settings

    all_c = []
    for key in settings.keys():
        sub_c = []
        for s_i in range(len(settings[key])):
            sub_c.append([[key] + list(x) for x in list(it.product(*settings[key][s_i]))])
        all_c.extend(sub_c)

    setup_collection = []
    training_options = list(it.product(*all_c))
    for variation in training_options:
        base_opt = copy.deepcopy(opt)
        base_d_opt = vars(base_opt)
        # WHY??? you never use base_d_opt again
        for i, sub_variation in enumerate(variation):
            base_d_opt[sub_variation[0]][sub_variation[1]] = sub_variation[2]
            base_d_opt['iter_idx'] = i
        setup_collection.append(base_opt)

        return setup_collection


def gimme_save_string(opt):
    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key],dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n\n'
    return base_str


class CSVlogger():
    def __init__(self, logname, header_names):
        self.header_names = header_names
        self.logname      = logname
        with open(logname,"a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(header_names)

    def write(self, inputs):
        with open(self.logname,"a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(inputs)


def write_submission(predictions, epoch, save_path):

    paths = glob.glob('/export/data/mdorkenw/data/alaska2/Test/*.jpg')
    IDs = [i[-(4 + 4):] for i in paths]
    submission = pd.DataFrame({'ID': IDs, 'Label': list(predictions)})
    filename = save_path + f'/submission/Submission_epoch_{epoch}.csv'
    submission.to_csv(filename, index=False)


def progress_plotter(x, train_loss, train_metric, labels, savename='result.svg'):
    plt.style.use('ggplot')
    f, ax = plt.subplots(1)
    ax.plot(x, train_loss, 'b--', label=labels[0])

    axt = ax.twinx()
    axt.plot(x, train_metric, 'b', label=labels[1])

    ax.legend(loc=0)
    axt.legend(loc=2)

    f.suptitle('Loss and Evaluation Metric Progression')
    f.set_size_inches(15, 10)
    f.savefig(savename)
    plt.close()


def summary_plots(loss_dic_train, loss_dic_test, epoch, save_path):
    progress_plotter(np.arange(0, len(loss_dic_train["Loss"])), loss_dic_train["Loss"], loss_dic_test["AUC"],
                     ["Loss Train", "AUC Test"], save_path + '/Loss_AUC.png')


class loss_tracking():
    def __init__(self, names):
        super(loss_tracking, self).__init__()
        self.loss_dic = names
        self.hist = {x: np.array([]) for x in self.loss_dic}
        self.keys = [*self.hist]

    def reset(self):
        self.dic = {x: np.array([]) for x in self.loss_dic}

    def append(self, losses):
        # assert (len(self.keys)-2 == len(losses))
        for idx in range(len(losses)):
            self.dic[self.keys[idx]] = np.append(self.dic[self.keys[idx]], losses[idx])

    def append_auc(self, auc):
        self.dic['AUC'] = np.append(self.dic['AUC'], auc)

    def append_binary_acc(self, acc):
        self.dic['binary_acc'] = np.append(self.dic['binary_acc'], acc)

    def append_accs(self, acss):
        for idx in range(len(acss)):
            self.dic[self.keys[idx + 2]] = np.append(self.dic[self.keys[idx+2]], acss[idx])

    def get_iteration_mean(self, num=20):
        mean = []
        for idx in range(2):
            if len(self.dic[self.keys[idx]]) <= num:
                mean.append(np.mean(self.dic[self.keys[idx]]))
            else:
                mean.append(np.mean(self.dic[self.keys[idx]][-num:]))
        return mean

    def get_mean(self):
        self.mean = []
        for idx in range(len(self.keys)):
            self.mean.append(np.mean(self.dic[self.keys[idx]]))
        self.history()

    def history(self):
        for idx in range(len(self.keys)):
            self.hist[self.keys[idx]] = np.append(self.hist[self.keys[idx]], self.mean[idx])

    def get_current_mean(self):
        return self.mean

    def get_hist(self):
        return self.hist


def auc_weighted(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        competition_metric += submetric

    return competition_metric / normalization


def auc(y_true, y_valid):
    return metrics.auc(y_true, y_valid)


def acc_per_class(pred, target, dic):
    n_classes = dic.Network['n_classes']
    accs = np.zeros(n_classes)
    for idx in range(n_classes):
        index = np.where(target == idx)[0]
        if len(index) > 0:
            accs[idx] = (pred[index] == idx).mean()
        else:
            accs[idx] = 1
    return accs
