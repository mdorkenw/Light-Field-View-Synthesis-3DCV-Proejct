import numpy as np, time, random, csv, glob
import torch, ast, pandas as pd, copy, itertools as it, os
import argparse
import itertools as it, copy
import torchvision
import matplotlib.pyplot as plt
plt.switch_backend('agg')


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


class Loss_Tracking():
    def __init__(self, names):
        super(Loss_Tracking, self).__init__()
        self.loss_dic = names
        self.hist = {x: np.array([]) for x in self.loss_dic}
        self.keys = [*self.hist]

    def reset(self):
        self.dic = {x: np.array([]) for x in self.loss_dic}

    def append(self, losses):
        for idx in range(len(losses)):
            self.dic[self.keys[idx]] = np.append(self.dic[self.keys[idx]], losses[idx])

    def get_iteration_mean(self, num=20):
        mean = []
        for idx in range(len(self.keys)):
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


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def save_images(recon, x3, opt, epoch, mode, num=5, folder='images'):
    x3 = x3.transpose(1,2)
    recon = recon.transpose(1, 2)
    for i in range(num):
        image = torch.cat([denorm(recon[i]).float().cpu().data,denorm(x3[i]).float().cpu().data])
        torchvision.utils.save_image(image, opt.Paths['save_path'] + '/'+folder+'/{:03d}_seq_'.format(epoch + 1) + mode + '_{:03d}'.format(i) + '.png', nrow=9)


def update_iter_train(data_iter, loss_track, epoch):
    """ Update data iterator description with current loss.
    """
    loss, loss_recon, loss_kl = loss_track.get_iteration_mean()
    inp_string = 'Epoch {} || Loss: {} | Loss_recon: {} | Loss_kl: {}'.format(epoch, np.round(loss, 3), np.round(loss_recon, 3), np.round(loss_kl, 3))
    data_iter.set_description(inp_string)


def update_iter_validate(data_iter, loss_track, epoch):
    """ Update data iterator description with current losses.
    """
    loss_hor, loss_vert, loss_diag, loss_diag_sum, _, _ = loss_track.get_iteration_mean()
    inp_string = 'Epoch {} || L_hor: {} | L_vert: {} | L_diag:{} | L_diag_sum:{}'.format(epoch, np.round(loss_hor, 3),
                                            np.round(loss_vert, 3), np.round(loss_diag, 3), np.round(loss_diag_sum, 3))
    data_iter.set_description(inp_string)


def get_all_images_from_dict(file_dict, dic):
    """ Get all directions from batch.
    """
    horizontal = file_dict["horizontal"].to(dic.Training['device'])
    horizontal_mask = file_dict["horizontal_mask"].to(dic.Training['device'])
    vertical = file_dict["vertical"].to(dic.Training['device'])
    vertical_mask = file_dict["vertical_mask"].to(dic.Training['device'])
    diagonal = file_dict["diagonal"].to(dic.Training['device'])
    diagonal_mask = file_dict["diagonal_mask"].to(dic.Training['device'])
    return horizontal, horizontal_mask, vertical, vertical_mask, diagonal, diagonal_mask


def get_crops(tensor):
    """ Crop large tensor into list of 48x48 patches with 16px overlap.
    """
    numb = int(tensor.shape[-1]/16-2)
    crops = list()
    for i in range(numb):
        for j in range(numb):
            crops.append(tensor[...,i*16:32+(i+1)*16,j*16:32+(j+1)*16])
    return crops


def crops_to_tensor(list_tensors):
    """ Reconstruct original image from list of crops.
    """
    numb = int(np.sqrt(len(list_tensors)))
    size = numb*16
    results = torch.zeros((3,9,size,size))
    for i, elem in enumerate(list_tensors):
        y = i%numb
        x = i//numb
        results[...,x*16:(x+1)*16,y*16:(y+1)*16] = elem[...,16:32,16:32]
    return results.transpose(0,1)


def crop_original_full_image(tensor):
    """ Crop border of original image.
    """
    return tensor[...,16:-16,16:-16]


def save_full_image_grid(original_, generated_, name, opt):
    """ Save original, generated images and differences.
    """
    original = denorm(crop_original_full_image(original_.transpose(0,1)))
    generated = denorm(generated_)
    
    difference = torch.abs(original-generated)
    difference_rep = torch.abs(original-torch.cat([original[4:5]]*9))
    comb = torch.cat([original, generated, difference, difference_rep])
    
    if opt.Misc['test_mode']: path = os.path.dirname(opt.Paths['save_path'] + opt.Paths['load_network_path'])+"/summary_plots/"
    else: path = opt.Paths['save_path']+"/summary_plots/"
    torchvision.utils.save_image(comb,path+name+".png",nrow=9)
