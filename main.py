import numpy as np, os, sys, pandas as pd, ast, time, gc
import torch, torch.nn as nn, pickle as pkl, random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm, trange
import network as net
import auxiliaries as aux
import loss as Loss
import dataloader as dloader
import argparse
from distutils.dir_util import copy_tree
from datetime import datetime

seed = 42
print(f'setting everything to seed {seed}')
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def trainer(network, dic, epoch, data_loader, loss_track, optimizer, loss_func, scheduler):

    _ = network.train()
    loss_track.reset()
    logits_collect = []; labels_collect = []
    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss: --- | Acc: ---'.format(epoch)
    data_iter.set_description(inp_string)
    for image_idx, file_dict in enumerate(data_iter):

        input = file_dict["input"].type(torch.FloatTensor).cuda()
        label = file_dict["label"].type(torch.FloatTensor).cuda()
        mask = file_dict["mask"].type(torch.FloatTensor).cuda()

        logits = network(input, mask)
        loss = loss_func(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## Compute Accuracy
        label = np.argmax(label.cpu().data.numpy(), axis=1)
        acc = (np.sum(np.argmax(logits.cpu().data.numpy(), axis=1) == label))/logits.size(0)

        ## Collect logits to compute weighted AUC later
        logits_collect.append(logits.detach().cpu())
        labels_collect.append(label)

        loss_dic = [loss.item(), acc]
        loss_track.append(loss_dic)

        if image_idx % 20 == 0:
            loss_mean, acc_mean, *_ = loss_track.get_iteration_mean()
            inp_string = 'Epoch {} || Loss: {} | Acc: {}'.format(epoch, np.round(loss_mean, 2), np.round(acc_mean, 3))
            data_iter.set_description(inp_string)

    logits = torch.cat(logits_collect, dim=0)
    label = np.concatenate(labels_collect, axis=0)

    ## compute weighted AUC
    prediction = np.argmax(logits.cpu().data.numpy(), axis=1)
    accs = aux.acc_per_class(prediction, label, dic)
    loss_track.append_accs(accs)

    if 5 > logits.size(1) > 1:
        pred = 1 - nn.functional.softmax(logits, dim=1).numpy()[:, 0]
        label = label.clip(min=0, max=1).astype(int)
    elif logits.size(1) > 5:
        pred = 1 - np.sum(nn.functional.softmax(logits, dim=1).numpy()[:, :3], axis=1)
        label[label<3]  = 0
        label[label>=3] = 1
    else:
        pred = 1 - logits.numpy().reshape(-1)
    
    auc = aux.auc_weighted(label, pred)
    loss_track.append_auc(auc)

    ## Compute binary classification
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    acc = (np.round(pred == label)).mean()
    loss_track.append_binary_acc(acc)

    ### Empty GPU cache
    torch.cuda.empty_cache()
    loss_track.get_mean()
    scheduler.step(loss_track.get_current_mean()[0])


def validator(network, dic, epoch, data_loader, loss_track, loss_func):

    _ = network.eval()
    loss_track.reset()

    logits_collect = []; labels_collect = []
    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss: --- | Acc: ---'.format(epoch)
    data_iter.set_description(inp_string)

    with torch.no_grad():
        for image_idx, file_dict in enumerate(data_iter):

            input = file_dict["input"].type(torch.FloatTensor).cuda()
            label = file_dict["label"].type(torch.FloatTensor).cuda()
            mask = file_dict["mask"].type(torch.FloatTensor).cuda()

            logits = network(input, mask)
            loss = loss_func(logits, label)

            ## Compute Accuracy
            label = np.argmax(label.cpu().data.numpy(), axis=1)
            acc = (np.sum(np.argmax(logits.cpu().data.numpy(), axis=1) == label))/logits.size(0)

            ## Collect logits to compute weighted AUC later
            logits_collect.append(logits.detach().cpu())
            labels_collect.append(label)

            loss_dic = [loss.item(), acc]
            loss_track.append(loss_dic)

            if image_idx % 20 == 0:
                loss_mean, acc_mean, *_ = loss_track.get_iteration_mean()
                inp_string = 'Epoch {} || Loss: {} | Acc: {}'.format(epoch, np.round(loss_mean, 2), np.round(acc_mean, 3))
                data_iter.set_description(inp_string)


    logits = torch.cat(logits_collect, dim=0)
    label = np.concatenate(labels_collect, axis=0)

    ## compute weighted AUC
    prediction = np.argmax(logits.cpu().data.numpy(), axis=1)
    accs = aux.acc_per_class(prediction, label, dic)
    loss_track.append_accs(accs)

    if 5 > logits.size(1) > 1:
        pred = 1 - nn.functional.softmax(logits, dim=1).numpy()[:, 0]
        label = label.clip(min=0, max=1).astype(int)
    elif logits.size(1) > 5:
        pred = 1 - np.sum(nn.functional.softmax(logits, dim=1).numpy()[:, :3], axis=1)
        label[label<3]  = 0
        label[label>=3] = 1
    else:
        pred = 1 - logits.numpy().reshape(-1)
    
    auc = aux.auc_weighted(label, pred)
    loss_track.append_auc(auc)

    ## Compute binary classification
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    acc = (np.round(pred == label)).mean()
    loss_track.append_binary_acc(acc)

    ### Empty GPU cache
    torch.cuda.empty_cache()
    loss_track.get_mean()


def tester(network, epoch, data_loader, save_path):

    _ = network.eval()
    logits_collect = []
    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} '.format(epoch)
    data_iter.set_description(inp_string)

    with torch.no_grad():
        for image_idx, file_dict in enumerate(data_iter):

            input = file_dict["input"].type(torch.FloatTensor).cuda()
            mask = file_dict["mask"].type(torch.FloatTensor).cuda()

            logits   = network(input, mask)
            logits_collect.append(logits.detach().cpu())

    logits = torch.cat(logits_collect, dim=0)

    if 5 > logits.size(1) > 1:
        pred = 1 - nn.functional.softmax(logits, dim=1).numpy()[:, 0]
    elif logits.size(1) > 5:
        pred = 1 - np.sum(nn.functional.softmax(logits, dim=1).numpy()[:, :3], axis=1)
    else:
        pred = 1 - logits.numpy().reshape(-1)

    aux.write_submission(pred, epoch, save_path)


def main(opt):
    """============================================"""
    ### Load Network
    network = net.Net(opt.Network).cuda()
    print("Number of parameters in model", sum(p.numel() for p in network.parameters()))

    ###### Define Optimizer ######
    loss_func   = Loss.LabelSmoothing(opt.Training)
    optimizer   = torch.optim.AdamW(network.parameters(), lr=opt.Training['lr'], weight_decay=opt.Training['weight_decay'])
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-8,
                                                             threshold=0.0001, threshold_mode='abs')
    ###### Create Dataloaders ######
    train_dataset     = dloader.dataset(opt, mode='train')
    train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers=opt.Training['workers'],
                                                    batch_size=opt.Training['bs'], shuffle=True)
    val_dataset       = dloader.dataset(opt, mode='evaluation')
    val_data_loader   = torch.utils.data.DataLoader(val_dataset, num_workers=opt.Training['workers'],
                                                    batch_size=opt.Training['bs'], shuffle=False)
    test_dataset       = dloader.dataset(opt, mode='test')
    test_data_loader   = torch.utils.data.DataLoader(test_dataset, num_workers=opt.Training['workers'],
                                                     batch_size=opt.Training['bs'], shuffle=False)

    ###### Set Logging Files ######
    dt = datetime.now()
    dt = '{}-{}-{}-{}-{}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    opt.Training['name'] = 'Model' + '_Date-' + dt  # +str(opt.iter_idx)+
    if opt.Training['savename'] != "":
        opt.Training['name'] += '_' + opt.Training['savename']

    save_path = opt.Paths['save_path'] + "/" + opt.Training['name']

    ### Make the saving directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        count = 1
        while os.path.exists(save_path):
            count += 1
            svn = opt.Training['name'] + "_" + str(count)
            save_path = opt.Paths['save_path'] + "/" + svn
        opt.Training['name'] = svn
        os.makedirs(save_path)
    opt.Paths['save_path'] = save_path

    def make_folder(name):
        if not os.path.exists(name):
            os.makedirs(name)

    # Make summary plots, images, segmentation and videos folder
    save_summary = save_path + '/summary_plots'
    make_folder(save_path + '/summary_plots')
    make_folder(save_path + '/submission')

    ### Copy Code !!
    copy_tree('./', save_path + '/code/')
    save_str = aux.gimme_save_string(opt)

    ### Save rudimentary info parameters to text-file and pkl.
    with open(opt.Paths['save_path'] + '/Parameter_Info.txt', 'w') as f:
        f.write(save_str)
    pkl.dump(opt, open(opt.Paths['save_path'] + "/hypa.pkl", "wb"))

    ## Loss tracker is implented in such a way that the first 2 elements are added every iteration
    Acc_classes = [f"Acc class {str(i)}" for i in range(opt.Network['n_classes'])]
    logging_keys = ["Loss", "Acc Mean", *Acc_classes, "binary_acc", "AUC"]

    loss_track_train = aux.loss_tracking(logging_keys)
    loss_track_test = aux.loss_tracking(logging_keys)

    ### Setting up CSV writers
    full_log_train = aux.CSVlogger(save_path + "/log_per_epoch_train.csv", ["Epoch", "Time", "LR"] + logging_keys)
    full_log_test = aux.CSVlogger(save_path + "/log_per_epoch_test.csv", ["Epoch", "Time", "LR"] + logging_keys)

    epoch_iterator = tqdm(range(0, opt.Training['n_epochs']), ascii=True, position=1)
    best_val_auc   = 0

    for epoch in epoch_iterator:
        epoch_time = time.time()

        ##### Training ########
        epoch_iterator.set_description("Training with lr={}".format(np.round([group['lr'] for group in optimizer.param_groups][0], 6)))
        trainer(network, opt, epoch, train_data_loader, loss_track_train, optimizer, loss_func, scheduler)

        ###### Validation #########
        epoch_iterator.set_description('Validating...')
        validator(network, opt, epoch, val_data_loader, loss_track_test, loss_func)

        ## Best Validation Score
        current_auc = loss_track_test.get_current_mean()[-1]
        if current_auc > best_val_auc:
            ## Forward pass on test set + write submission csv file
            epoch_iterator.set_description('Testing...')
            tester(network, epoch, test_data_loader, opt.Paths['save_path'])
            ###### SAVE CHECKPOINTS ########
            save_dict = {'epoch': epoch+1, 'state_dict': network.state_dict(), 'optim_state_dict': optimizer.state_dict()}
            torch.save(save_dict, opt.Paths['save_path'] + '/checkpoint_best_val.pth.tar')
            best_val_auc = current_auc

        ###### Logging Epoch Data ######]
        epoch_time =  time.time() - epoch_time
        full_log_train.write([epoch, epoch_time, [group['lr'] for group in optimizer.param_groups][0], *loss_track_train.get_current_mean()])
        full_log_test.write([epoch, epoch_time, [group['lr'] for group in optimizer.param_groups][0], *loss_track_test.get_current_mean()])

        ###### Generating Summary Plots #######
        # aux.summary_plots(loss_track_train.get_hist(), loss_track_test.get_hist(), epoch, save_summary)
        _ = gc.collect()


### Start Training ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config", type=str, default='/export/home/mdorkenw/code/ALASKA2/network_base_setup.txt',
                        help="Define config file")
    args = parser.parse_args()
    training_setups = aux.extract_setup_info(args.config)

    # find all the gpus we want to use now (currently not necessary, only if we want to run different
    # training setting on different gpus at the same time)
    gpus = []
    for tr in training_setups:
        for GPU in tr.Training['GPU']:
            gpus.append(str(GPU))

    gpus = ",".join(gpus)

    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    for training_setup in tqdm(training_setups, desc='Training Setups... ', position=0, ascii=True):
        main(training_setup)
