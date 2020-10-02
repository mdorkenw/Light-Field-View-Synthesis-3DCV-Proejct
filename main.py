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


def trainer(network, dic, epoch, data_loader, loss_track, optimizer, loss_func, scheduler, use_scheduler):

    _ = network.train()
    loss_track.reset() # Start loss track
    data_iter = tqdm(data_loader, position=2)
    data_iter.set_description('Epoch {} || Loss: --- | Loss_recon: --- | Loss_kl: ---'.format(epoch))
    
    for image_idx, file_dict in enumerate(data_iter):

        x = file_dict["x"].to(dic.Training['device'])
        x_mask = file_dict["x_mask"].to(dic.Training['device'])

        img_recon, mu, covar = network(x)
        loss, loss_recon, loss_kl = loss_func(img_recon*x_mask, x*x_mask, mu, covar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_track.append([loss.item(), loss_recon.item(), loss_kl.item()]) # Save loss
        if image_idx % 20 == 0: aux.update_iter_train(data_iter, loss_track, epoch) # Update description
            
    #aux.update_iter_train(data_iter, loss_track, epoch) # Update description
    loss_track.get_mean() # Finish loss track

    aux.save_images(img_recon, x, dic, epoch, 'train') # Save images
    if torch.cuda.is_available(): torch.cuda.empty_cache() # Empty GPU cache
    if use_scheduler: scheduler.step(loss_track.get_current_mean()[0]) # Scheduler step


def validator(network, dic, epoch, data_loader, loss_track, loss_func):

    _ = network.eval()
    loss_track.reset() # Start loss track
    data_iter = tqdm(data_loader, position=2)
    data_iter.set_description('Epoch {} || Loss: --- | Loss_recon: --- | Loss_kl: ---'.format(epoch))
    
    with torch.no_grad():
        for image_idx, file_dict in enumerate(data_iter):

            x = file_dict["x"].to(dic.Training['device'])
            x_mask = file_dict["x_mask"].to(dic.Training['device'])

            img_recon, mu, covar = network(x)
            loss, loss_recon, loss_kl = loss_func(img_recon*x_mask, x*x_mask, mu, covar)

            loss_track.append([loss.item(), loss_recon.item(), loss_kl.item()]) # Save loss
            if image_idx % 9 == 0: aux.update_iter_train(data_iter, loss_track, epoch) # Update description
    
    #aux.update_iter_train(data_iter, loss_track, epoch) # Update description
    loss_track.get_mean() # Finish loss track
    
    aux.save_images(img_recon, x, dic, epoch, 'test') # Save images
    if torch.cuda.is_available(): torch.cuda.empty_cache() # Empty GPU cache


def validator_full(network, dic, epoch, data_loader, loss_track, loss_func):
    """ Uses diagonal mask for all directions to make losses comparable.
        Therefore, the loss here will probably be lower that during training.
        Does not yet work correctly for VAE!!!
    """

    _ = network.eval()
    loss_track.reset() # Start loss track
    data_iter = tqdm(data_loader, position=2)
    data_iter.set_description('Epoch {} || L_hor: --- | L_vert: --- | L_diag: --- | L_diag_sum: ---'.format(epoch))
    
    with torch.no_grad():
        for image_idx, file_dict in enumerate(data_iter):
            # Inputs
            horizontal, horizontal_mask, vertical, vertical_mask, diagonal, diagonal_mask = aux.get_all_images_from_dict(file_dict, dic)
            # Latent representations
            latent_hor, _, _ = network.encode_reparametrize(horizontal)
            latent_vert, _, _ = network.encode_reparametrize(vertical)
            latent_diag, _, _ = network.encode_reparametrize(diagonal)
            latent_diag_sum = (latent_hor+latent_vert)/2
            # Diffrence of latent representations
            latent_difference_hor_vert = loss_func.reconstruction_loss(latent_hor, latent_vert)
            latent_difference_diag = loss_func.reconstruction_loss(latent_diag, latent_diag_sum)
            # Loss hor, vert
            img_recon_hor, img_recon_vert = network.decode(latent_hor), network.decode(latent_vert)
            loss_hor = loss_func.reconstruction_loss(img_recon_hor*diagonal_mask, horizontal*diagonal_mask)
            loss_vert = loss_func.reconstruction_loss(img_recon_vert*diagonal_mask, vertical*diagonal_mask)
            # Loss diag, diag_sum
            img_recon_diag, img_recon_diag_sum = network.decode(latent_diag), network.decode(latent_diag_sum)
            loss_diag = loss_func.reconstruction_loss(img_recon_diag*diagonal_mask, diagonal*diagonal_mask)
            loss_diag_sum = loss_func.reconstruction_loss(img_recon_diag_sum*diagonal_mask, diagonal*diagonal_mask)
            
            loss_track.append([loss_hor.item(), loss_vert.item(), loss_diag.item(), loss_diag_sum.item(),
                                latent_difference_hor_vert.item(), latent_difference_diag.item()]) # Save loss
            
            if image_idx % 9 == 0: aux.update_iter_validate(data_iter, loss_track, epoch) # Update description
    
    #aux.update_iter_validate(data_iter, loss_track, epoch) # Update description
    loss_track.get_mean() # Finish loss track
    
    aux.save_images(img_recon_hor, horizontal, dic, epoch, 'validate_hor', folder="images_validate")
    aux.save_images(img_recon_vert, vertical, dic, epoch, 'validate_vert', folder="images_validate")
    aux.save_images(img_recon_diag, diagonal, dic, epoch, 'validate_diag', folder="images_validate")
    aux.save_images(img_recon_diag_sum, diagonal, dic, epoch, 'validate_diag_sum', folder="images_validate")
    
    if torch.cuda.is_available(): torch.cuda.empty_cache() # Empty GPU cache


def main(opt):
    """============================================"""
    
    seed = opt.Training['global_seed']
    print(f'\nSetting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    ### Create Network
    network = net.VAE(opt.Network).to(opt.Training['device'])
    if opt.Network['load_trained']:
        save_dict = torch.load(opt.Paths['save_path'] + opt.Paths['load_network_path'])
        network.load_state_dict(save_dict['state_dict'])
        print('Loaded model from '+opt.Paths['load_network_path'])
        

    ###### Define Optimizer ######
    loss_func   = Loss.Loss(opt.Training).to(opt.Training['device'])
    optimizer   = torch.optim.Adam(network.parameters(), lr=opt.Training['lr'], weight_decay=opt.Training['weight_decay'])
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.Training['sched_factor'], patience=opt.Training['sched_patience'], min_lr=1e-8,
                                                             threshold=0.0001, threshold_mode='abs')

    ###### Create Dataloaders ######
    train_dataset        = dloader.dataset(opt, mode='train')
    train_data_loader    = torch.utils.data.DataLoader(train_dataset, num_workers=opt.Training['workers'],batch_size=opt.Training['bs'])
    test_dataset         = dloader.dataset(opt, mode='test', return_mode='all' if opt.Misc['use_full_validate'] else '')
    test_data_loader     = torch.utils.data.DataLoader(test_dataset, num_workers=opt.Training['workers'],batch_size=opt.Training['bs'])

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
    if opt.Misc['use_full_validate']: Path(save_path + '/images_validate').mkdir(parents=True, exist_ok=True)

    ### Copy Code !!
    if opt.Misc["copy_code"]: copy_tree('./', save_path + '/code/') # Does not work for me, I think the paths are too long for windows
    save_str = aux.gimme_save_string(opt)

    ### Save rudimentary info parameters to text-file and pkl.
    with open(opt.Paths['save_path'] + '/Parameter_Info.txt', 'w') as f:
        f.write(save_str)
    pkl.dump(opt, open(opt.Paths['save_path'] + "/hypa.pkl", "wb"))

    ## Loss tracker is implented in such a way that the first 2 elements are added every iteration
    logging_keys = ["Loss", "L_recon", 'L_kl']
    logging_keys_test = ["L_recon_hor", "L_recon_vert", "L_recon_diag", "L_recon_diag_sum", "D_hor_vert", "D_diag"] if opt.Misc['use_full_validate'] else logging_keys

    loss_track_train = aux.Loss_Tracking(logging_keys)
    loss_track_test = aux.Loss_Tracking(logging_keys_test)

    ### Setting up CSV writers
    full_log_train = aux.CSVlogger(save_path + "/log_per_epoch_train.csv", ["Epoch", "Time", "LR"] + logging_keys)
    full_log_test = aux.CSVlogger(save_path + "/log_per_epoch_test.csv", ["Epoch", "Time", "LR"] + logging_keys_test)

    epoch_iterator = tqdm(range(0, opt.Training['n_epochs']), ascii=True, position=1)
    best_loss = np.inf

    for epoch in epoch_iterator:
        epoch_time = time.time()

        ##### Training ########
        epoch_iterator.set_description("Training with lr={}".format(np.round([group['lr'] for group in optimizer.param_groups][0], 6)))
        trainer(network, opt, epoch, train_data_loader, loss_track_train, optimizer, loss_func, scheduler, opt.Training['use_sched'])

        ###### Validation #########
        epoch_iterator.set_description('Validating...')
        if opt.Misc['use_full_validate']:
            validator_full(network, opt, epoch, test_data_loader, loss_track_test, loss_func)
        else:
            validator(network, opt, epoch, test_data_loader, loss_track_test, loss_func)

        ## Best Validation Loss
        current_loss = loss_track_test.get_current_mean()[0]
        if current_loss < best_loss:
            ###### SAVE CHECKPOINTS ########
            save_dict = {'epoch': epoch+1, 'state_dict': network.state_dict(), 'optim_state_dict': optimizer.state_dict()}
            torch.save(save_dict, opt.Paths['save_path'] + '/checkpoint_best_val.pth.tar')
            best_loss = current_loss
        
        ## Always save occasionally
        if epoch % opt.Training['save_every'] == 0:
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

    for training_setup in tqdm(training_setups, desc='Training Setups... ', position=0, ascii=True):
        gpu = training_setup.Training['GPU'] if torch.cuda.is_available() else []
        device = torch.device('cuda:{}'.format(gpu[0])) if gpu and torch.cuda.is_available() else torch.device('cpu')
        training_setup.Training['device'] = device
        training_setup.Network['device'] = device
        
        training_setup.Training['use_kl'] = training_setup.Network['use_VAE']
        
        main(training_setup)
