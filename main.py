import numpy as np, os, sys, pandas as pd, ast, time, gc
import torch, torch.nn as nn, pickle as pkl, random
from tqdm import tqdm, trange
import network as net
import auxiliaries as aux
import loss as Loss
import dataloader as dloader
import argparse
from distutils.dir_util import copy_tree
from datetime import datetime
from pathlib import Path


def trainer(network, dic, epoch, data_loader, loss_track, optimizer, loss_func, scheduler):

    _ = network.train()
    loss_track.reset()
    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss: --- | Acc: ---'.format(epoch)
    data_iter.set_description(inp_string)
    for image_idx, file_dict in enumerate(data_iter):

        x = file_dict["x"].type(torch.FloatTensor).cuda().transpose(1, 2)

        img_recon, mu, covar = network(x)
        loss, loss_recon, loss_kl = loss_func(img_recon, x, mu, covar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_dic = [loss.item(), loss_recon.item(), loss_kl.item()]
        loss_track.append(loss_dic)

        if image_idx % 20 == 0:
            _, loss_recon, loss_kl = loss_track.get_iteration_mean()
            inp_string = 'Epoch {} || Loss: {} | Loss_kl: {}'.format(epoch, np.round(loss_recon, 3), np.round(loss_kl, 3))
            data_iter.set_description(inp_string)

    ## Save images
    aux.save_images(img_recon, x, dic, epoch, 'train')
    ### Empty GPU cache
    torch.cuda.empty_cache()
    loss_track.get_mean()
    scheduler.step(loss_track.get_current_mean()[0])


def validator(network, dic, epoch, data_loader, loss_track, loss_func):

    _ = network.eval()
    loss_track.reset()
    data_iter = tqdm(data_loader, position=2)
    inp_string = 'Epoch {} || Loss: --- | Acc: ---'.format(epoch)
    data_iter.set_description(inp_string)
    with torch.no_grad():
        for image_idx, file_dict in enumerate(data_iter):

            x = file_dict["x"].type(torch.FloatTensor).cuda().transpose(1, 2)

            img_recon, mu, covar = network(x)
            loss, loss_recon, loss_kl = loss_func(img_recon, x, mu, covar)

            loss_dic = [loss.item(), loss_recon.item(), loss_kl.item()]
            loss_track.append(loss_dic)

            if image_idx % 10 == 0:
                _, loss_recon, loss_kl = loss_track.get_iteration_mean()
                inp_string = 'Epoch {} || Loss: {} | Loss_kl: {}'.format(epoch, np.round(loss_recon, 3), np.round(loss_kl, 3))
                data_iter.set_description(inp_string)

    ## Save images
    aux.save_images(img_recon, x, dic, epoch, 'test')

    ### Empty GPU cache
    torch.cuda.empty_cache()
    loss_track.get_mean()

def main(opt):
    """============================================"""
    
    seed = 42
    print(f'\nsetting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    ### Create Network
    network = net.VAE(opt.Network).cuda()

    ###### Define Optimizer ######
    loss_func   = Loss.Loss(opt.Training)
    optimizer   = torch.optim.Adam(network.parameters(), lr=opt.Training['lr'], weight_decay=opt.Training['weight_decay'])
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.Training['sched_factor'], patience=opt.Training['sched_patience'], min_lr=1e-8,
                                                             threshold=0.0001, threshold_mode='abs')

    ###### Create Dataloaders ######
    train_dataset     = dloader.dataset(opt, mode='train')
    train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers=opt.Training['workers'],
                                                    batch_size=opt.Training['bs'], shuffle=True)
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

    # Make summary plots, images, segmentation and videos folder
    save_summary = save_path + '/summary_plots'
    Path(save_path + '/summary_plots').mkdir(parents=True, exist_ok=True)
    Path(save_path + '/images').mkdir(parents=True, exist_ok=True)

    ### Copy Code !!
    if opt.Misc["copy_code"]: copy_tree('./', save_path + '/code/') # Does not work for me, I think the paths are too long for windows
    save_str = aux.gimme_save_string(opt)

    ### Save rudimentary info parameters to text-file and pkl.
    with open(opt.Paths['save_path'] + '/Parameter_Info.txt', 'w') as f:
        f.write(save_str)
    pkl.dump(opt, open(opt.Paths['save_path'] + "/hypa.pkl", "wb"))

    ## Loss tracker is implented in such a way that the first 2 elements are added every iteration
    logging_keys = ["Loss", "L_recon", 'L_kl']

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
        validator(network, opt, epoch, test_data_loader, loss_track_test, loss_func)

        ## Best Validation Score
        current_auc = loss_track_test.get_current_mean()[0] # Was [-1], but should be [0]?
        if current_auc > best_val_auc:
            ###### SAVE CHECKPOINTS ########
            save_dict = {'epoch': epoch+1, 'state_dict': network.state_dict(), 'optim_state_dict': optimizer.state_dict()}
            torch.save(save_dict, opt.Paths['save_path'] + '/checkpoint_best_val.pth.tar')
            best_val_auc = current_auc
        
        ## Always save occasionally
        if epoch % opt.Training['save_every']:
            ###### SAVE CHECKPOINTS ########
            save_dict = {'epoch': epoch+1, 'state_dict': network.state_dict(), 'optim_state_dict': optimizer.state_dict()}
            torch.save(save_dict, opt.Paths['save_path'] + '/checkpoint_epoch_{}.pth.tar'.format(epoch))

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
    parser.add_argument("-cf", "--config", type=str, default='/export/home/mdorkenw/code/Lightfield/network_base_setup.txt',
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
