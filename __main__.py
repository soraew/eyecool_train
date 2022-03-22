from distutils.log import debug
from email.policy import default
import os
import logging
from datetime import datetime
import argparse
from pickletools import optimize
import shutil
# from tkinter.messagebox import NO
import zipfile
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import torch
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import albumentations as A

import sys
sys.path.append('.../NIR-ISL2021master/')
from datasets import eyeDataset
from models import EfficientUNet
from loss import Make_Criterion
from evaluation import evaluate_loc



if torch.cuda.is_available():
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    

experiment_name = 'M1-e5UNet'
dataset_name = 'CASIA-Iris-Mobile-V1.0'
assert dataset_name in ['CASIA-Iris-Africa','CASIA-distance', 'Occlusion', 'Off_angle', 'CASIA-Iris-Mobile-V1.0']


def get_args():
    parser = argparse.ArgumentParser(description='Train paprmeters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-e', '--epochs', type=int, default=96, dest='epoch_num')
    # parser.add_argument('-b', '--batch-size', type=int, nargs='?', default=8, dest='batch_size')
    # parser.add_argument('-l', '--learning-rate', type=float, nargs='?', default=0.002, dest='lr')
    # parser.add_argument('--log', type=str, default='logging.log', dest='log_name')
    # parser.add_argument('--ckp', type=str, default=None, help='load a pertrain model from .../xxx.pth', dest='checkpoints')
    # parser.add_argument('--gpu-ids', type=int, nargs='*', help='use cuda', dest='gpu_ids')
    parser.add_argument('--input', default=None)
    parser.add_argument('--input0', default=None) # this should be nucdre_zip
    parser.add_argument('--input1', default=None) # this should be jsons_zip
    parser.add_argument('--debug', default=False, type= bool) 
    parser.add_argument('--epochs', default=False, type=int) # 10 for on ahub
    parser.add_argument('--batchsize', default=8, type=int) # 2 for on mac, 8 for on ahub
    parser.add_argument("--output", type = Path)
    parser.add_argument("--tempDir", type = Path) 
    return parser.parse_args()


def main(train_args, data_root, debug):
    ########################################### logging and writer #############################################
    writer = SummaryWriter(log_dir=os.path.join(log_path, 'summarywriter_'+train_args['log_name'].split('.')[0]), comment=train_args['log_name'])

    logging.info('------------------------------------------------train configs------------------------------------------------')
    logging.info(train_args)

    ############################################# define a CNN #################################################
    # changed num_classes from 3 to 2
    net = EfficientUNet(num_classes=2).to(device)
    # print("net loc size : ", sys.getsizeof(net)/1e6) #size of nn
    if train_args['checkpoints']:
        net.load_state_dict(torch.load(train_args['checkpoints']))
    if train_args['gpu_ids']:
        net = torch.nn.DataParallel(net, device_ids=train_args['gpu_ids'])

    ########################################### dataset #############################################
    train_augment = A.Compose([
        # A.Resize(320, 544), # for Africa dataset
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.4),
        A.ShiftScaleRotate(p=0.4),
        A.Equalize(p=0.2)
    ])
    val_augment = A.Compose([
        # A.Resize(320, 544) # for Africa dataset
    ])

    train_dataset = eyeDataset(dataset_name, data_root=data_root, mode="train", transform=train_augment)
    val_dataset = eyeDataset(dataset_name, data_root=data_root, mode='val', transform=val_augment)
    train_loader = DataLoader(train_dataset, batch_size=train_args['batch_size'], num_workers=8, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_args['batch_size'], num_workers=8, drop_last=True)

    logging.info(f'data_augment are: \n {train_augment} \n {val_augment}')
    logging.info(f'The dataset {dataset_name} is ready!')

    ########################################### criterion #############################################
    criterion = Make_Criterion(deep_supervise = 1)
    # heatmap_criteria = torch.nn.MSELoss().to(device)
    # logging.info(f'''criterion is ready! \n{criterion} \n{heatmap_criteria}''')

    ########################################### optimizer #############################################
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * train_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': train_args['lr'], 'weight_decay': 1e-8}
    ], betas=(0.95, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=8, min_lr=1e-10, verbose=True)
    logging.info(f'optimizer is ready! \n{optimizer}')

    ######################################### train and val ############################################
    net.train()
    try:
        curr_epoch = 1
        train_args['best_record_inner'] = {'epoch': 0, 'val_loss': 999, 'E1': 999, 'IoU': 0, 'Dice': 0}
        train_args['best_record_outer'] = {'epoch': 0, 'val_loss': 999, 'E1': 999, 'IoU': 0, 'Dice': 0}

        for epoch in range(curr_epoch, train_args['epoch_num'] + 1):
            # train
            train(writer, train_loader, net, criterion, optimizer, epoch, train_args, debug)
            # validate
            val_loss = validate(writer, val_loader, net, criterion, optimizer, epoch, train_args, debug)
            # step scheduler
            scheduler.step(val_loss)

        writer.close()
        logging.info('-------------------------------------------------best record------------------------------------------------')
        logging.info('outer   epoch:{}  val loss {:.5f}  E1:{:.5f}   IoU:{:.5f}   Dice:{:.5f}'.format(
            train_args['best_record_outer']['epoch'], train_args['best_record_outer']['val_loss'], train_args['best_record_outer']['E1'],
            train_args['best_record_outer']['IoU'], train_args['best_record_outer']['Dice']
            ))
        logging.info('inner   epoch:{}  val loss {:.5f}  E1:{:.5f}   IoU:{:.5f}   Dice:{:.5f}'.format(
            train_args['best_record_inner']['epoch'], train_args['best_record_inner']['val_loss'], train_args['best_record_inner']['E1'],
            train_args['best_record_inner']['IoU'], train_args['best_record_inner']['Dice']
            ))

    except KeyboardInterrupt:
        print("interrupted")
        torch.save(net.module.state_dict(), log_path+'/INTERRUPTED.pth')
        logging.info('Saved interrupt!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


def train(writer, train_loader, net, criterion, optimizer, epoch, train_args, debug):
    logging.info('--------------------------------------------------training...------------------------------------------------')
    iters = len(train_loader)
    curr_iter = (epoch - 1) * iters
    print('--------------------------------------------------training...------------------------------------------------')
    for i, data in enumerate(train_loader):
        # print("data enumerated : ", i)
        image, iris_mask, pupil_mask = \
            data['image'], data['iris_mask'], data['pupil_mask'] #BCHW
            # deleted mask, iris_edge_mask -> iris_mask, pupil_edge_mask -> pupil_mask, heatmap
        
        # assert image.size()[2:] == mask.size()[2:]
        image = Variable(image).to(device)
        # mask = Variable(mask).to(device)
        iris_mask = Variable(iris_mask).to(device)
        pupil_mask = Variable(pupil_mask).to(device)
        # heatmap = Variable(heatmap).to(device)

        # WE CALL ZERO_GRAD FOR EVERY MINI-BATCH, BECAUSE PYTOCH ACCUMULATES GRADIENTS BY DEFAULT
        optimizer.zero_grad()

        # print("size of image : ", image.shape)
        outputs = net(image) # Non-empty 4D data tensor expected but got a tensor with sizes [1, 0, 276, 276]
        # deleted pred_mask, pred_heatmap
        pred_iris_mask, pred_pupil_mask, pred_heatmap = \
            outputs['pred_iris_mask'], outputs['pred_pupil_mask'], outputs['heatmap']
    

        # loss_mask = criterion(pred_mask, mask)
        loss_iris = criterion(pred_iris_mask, iris_mask)
        loss_pupil = criterion(pred_pupil_mask, pupil_mask)

        ################## heatmaps (we can output this later, maybe?) ##################
        # heatmap0 = transforms.Resize((pred_heatmap[0].size()[2:]))(heatmap)
        # heatmap1 = transforms.Resize((pred_heatmap[1].size()[2:]))(heatmap)
        # heatmap2 = transforms.Resize((pred_heatmap[2].size()[2:]))(heatmap)
        # loss_heatmap = heatmap_criteria(pred_heatmap[0], heatmap0) + heatmap_criteria(pred_heatmap[1], heatmap1) + heatmap_criteria(pred_heatmap[2], heatmap2)
        
        # I would like to plot some images and heatmaps 
        # print("shape of heatmap0 >> ", pred_heatmap[0].shape)
        # print("shape of heatmap1 >> ", pred_heatmap[1].shape)
        # print("shape of heatmap2 >> ", pred_heatmap[2].shape)
        # loss = loss_mask + loss_iris + 2*loss_pupil + 0.3*loss_heatmap
        loss = loss_iris + 2*loss_pupil 

        loss.backward()
        optimizer.step()

        writer.add_scalar('train_loss/iter', loss.item(), curr_iter)
        # writer.add_scalar('train_loss_mask/iter', loss_mask.item(), curr_iter)
        writer.add_scalar('train_loss_iris/iter', loss_iris.item(), curr_iter)
        writer.add_scalar('train_loss_pupil/iter', loss_pupil.item(), curr_iter)
        # writer.add_scalar('train_loss_heatmap/iter', loss_heatmap.item(), curr_iter)

        if (i + 1) % train_args['print_freq'] == 0:
            # removed losses that we don't need
            print('epoch:{:2d}  iter/iters:{:3d}/{:3d} iter%{:.5f} train_loss:{:.9f}   loss_iris:{:.9}   loss_pupil:{:.9}'.format(
                epoch, i+1, iters, (float(i+1)/float(iters)), loss, loss_iris, loss_pupil))
            logging.info('epoch:{:2d}  iter/iters:{:3d}/{:3d}  train_loss:{:.9f}  loss_iris:{:.9}   loss_pupil:{:.9}'.format(
                epoch, i+1, iters, loss, loss_iris, loss_pupil))

        curr_iter += 1

        # added this for debugging(one data point)
        # if i > 1:
        if debug:
            break



def validate(writer, val_loader, net, criterion, optimizer, epoch, train_args, debug):
    net.eval()
    print('--------------------------------------------------validating...------------------------------------------------')
    # e1, iou, dice = 0, 0, 0
    iris_e1, iris_dice, iris_iou, iris_tp, iris_fp, iris_fn = 0, 0, 0, 0, 0, 0
    iris_recall, iris_precision, pupil_recall, pupil_precision = 0, 0, 0, 0
    pupil_e1, pupil_dice, pupil_iou, pupil_tp, pupil_fp, pupil_fn = 0, 0, 0, 0, 0, 0
    # iris_hsdf, pupil_hsdf = 0, 0 # calculate hausdorff distance takes too long

    L = len(val_loader) # len(dataloader) = num_batches
    val_loss_ = 0
    val_loss_pupil = 0
    val_loss_iris = 0
    for data in val_loader:
        ###################################################################################################################################################
        ############ removed edge and heatmap data ##########
        ###################################################################################################################################################
        image, iris_mask, pupil_mask = \
            data['image'], data['iris_mask'], data['pupil_mask']

        # print(f"shape of each in validate: \n\
        #        image >> {image.shape}, iris_mask >> {iris_mask.shape}, pupil_mask >> {pupil_mask}")
   
        image = Variable(image).to(device)
        # mask = Variable(mask).to(device)
        # iris_edge = Variable(iris_edge).to(device)
        # pupil_edge = Variable(pupil_edge).to(device)
        iris_mask = Variable(iris_mask).to(device)
        pupil_mask = Variable(pupil_mask).to(device)
        
        with torch.no_grad():
            outputs = net(image)

        pred_iris_mask, pred_pupil_mask = \
            outputs['pred_iris_mask'], outputs['pred_pupil_mask']

        # loss_mask = criterion(pred_mask, mask)
        loss_iris = criterion(pred_iris_mask, iris_mask)
        val_loss_iris += loss_iris
        loss_pupil = criterion(pred_pupil_mask, pupil_mask)
        val_loss_pupil += loss_pupil
        val_loss = loss_iris + loss_pupil        
        val_loss_ += (loss_iris + 2*loss_pupil)

        # pred_iris_circle_mask, pred_iris_edge, _ = get_edge(pred_iris_mask)
        # pred_pupil_circle_mask, pred_pupil_egde, _ = get_edge(pred_pupil_mask) 

        #################### val for iris mask ##############
        iris_val_results = evaluate_loc(pred_iris_mask, iris_mask)  
        iris_e1 += iris_val_results["E1"]/L
        iris_dice += iris_val_results['Dice']/L
        iris_iou += iris_val_results['IoU']/L

        iris_tp += iris_val_results["TP"]#/L
        iris_fp += iris_val_results["FP"]#/L
        iris_fn += iris_val_results["FN"]#/L

        iris_recall += iris_val_results["recall"]/L
        iris_precision += iris_val_results["precision"]/L

        ################### val for pupil mask #############
        pupil_val_results = evaluate_loc(pred_pupil_mask, pupil_mask)
        pupil_e1 += pupil_val_results['E1']/L
        pupil_dice += pupil_val_results['Dice']/L  
        pupil_iou += pupil_val_results['IoU']/L  
        
        pupil_tp += pupil_val_results["TP"]#/L
        pupil_fp += pupil_val_results["FP"]#/L
        pupil_fn += pupil_val_results["FN"]#/L

        pupil_recall += pupil_val_results["recall"]/L
        pupil_precision += pupil_val_results["precision"]/L

        if debug:
            break
    val_loss_ /= L
    val_loss_iris /= L
    val_loss_pupil /= L

    print(f"Validation loss >>> {val_loss_}")
    print(f"Validation iris >>> tot_val_nums : {L} iris_val : e1 => {iris_e1}, dice => {iris_dice}, iou => {iris_iou}, tp => {iris_tp}, fp => {iris_fp}, fn => {iris_fn}, val_loss => {val_loss_iris}, recall => {iris_recall}, precision => {iris_precision}")
    print(f"Validation pupil >>> tot_val_nums : {L} pupil_val : e1 => {pupil_e1}, dice => {pupil_dice}, iou => {pupil_iou}, tp => {pupil_tp}, fp => {pupil_fp}, fn => {pupil_fn}, val_loss_pupil => {val_loss_pupil}, recall => {pupil_recall}, precision => {pupil_precision}")
        
    logging.info('------------------------------------------------current val result-----------------------------------------------')    
    # logging.info('>iris      epoch:{:2d}   val loss:{:.7f}   E1:{:.7}    Dice:{:.7f}   IOU:{:.7f}'. \
    #     format(epoch, loss_iris, iris_e1, iris_dice, iris_iou))
    # logging.info('>pupil     epoch:{:2d}   val loss:{:.7f}   E1:{:.7}    Dice:{:.7f}   IOU:{:.7f}'. \
    #     format(epoch, loss_pupil, pupil_e1, pupil_dice, pupil_iou))
    logging.info(f"epoch {epoch} iris >>> e1 => {iris_e1}, dice => {iris_dice}, iou => {iris_iou}, tp => {iris_tp}, fp => {iris_fp}, fn => {iris_fn}, recall => {iris_recall}, precision => {iris_precision}")
    logging.info(f"epoch {epoch} pupil >>> e1 => {pupil_e1}, dice => {pupil_dice}, iou => {pupil_iou}, tp => {pupil_tp}, fp => {pupil_fp}, fn => {pupil_fn}, recall => {pupil_recall}, precision => {pupil_precision}")
    

    ######################## adding scalars for log(summary writer) #########################
    writer.add_scalar('val_loss', val_loss_, epoch) # changed this from val_loss
    writer.add_scalar('val_loss_iris/iters', val_loss_iris, epoch)
    writer.add_scalar('val_loss_pupil/iters', val_loss_pupil, epoch)

    writer.add_scalar('iris_e1', iris_e1, epoch)
    writer.add_scalar('iris_dice', iris_dice, epoch)
    writer.add_scalar('iris_iou', iris_iou, epoch)
    writer.add_scalar('iris_recall', iris_recall, epoch)
    writer.add_scalar('iris_precision', iris_precision, epoch)

    writer.add_scalar('pupil_e1', pupil_e1, epoch)
    writer.add_scalar('pupil_dice', pupil_dice, epoch)
    writer.add_scalar('pupil_iou', pupil_iou, epoch)
    writer.add_scalar('pupil_recall', pupil_recall, epoch)
    writer.add_scalar('pupil_precision', pupil_precision, epoch)

    writer.add_scalar('lr', optimizer.param_groups[1]['lr'], epoch)

    ###################### adding images to summary writer ###########################
    writer.add_images('image', image, epoch)
    writer.add_images('iris_mask', iris_mask, epoch)
    writer.add_images('pred_iris_mask', pred_iris_mask>0, epoch)
    writer.add_images('pupil_mask', pupil_mask, epoch)
    writer.add_images('pred_pupil_mask', pred_pupil_mask>0, epoch)
    # writer.add_images('heatmap0', )
    
    ###################### getting best results for pupil(inner), iris(outer), respectively and saving them ###############
    if iris_e1 < train_args['best_record_outer']['E1']:
        train_args['best_record_outer']['epoch'] = epoch
        train_args['best_record_outer']['E1'] = iris_e1
        train_args['best_record_outer']['IoU'] = iris_iou
        train_args['best_record_outer']['Dice'] = iris_dice
        if train_args['gpu_ids']:
            torch.save(net.module.state_dict(), os.path.join(checkpoint_path, 'for_outer.pth'))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, 'optimizer_outer.pt'))
        else:
            torch.save(net.state_dict(), os.path.join(checkpoint_path, 'for_outer.pth')) 
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, 'optimizer_outer.pt'))
            ##########
        outer_checkpoints_name = 'epoch_%d_e1_%.7f_iou_%.7f_dice_%.7f' % (epoch, iris_e1, iris_iou, iris_dice)
        logging.info(f'Saved iris checkpoints {outer_checkpoints_name}.pth!')

    if pupil_e1 < train_args['best_record_inner']['E1']:
        train_args['best_record_inner']['epoch'] = epoch
        train_args['best_record_inner']['E1'] = pupil_e1
        train_args['best_record_inner']['IoU'] = pupil_iou
        train_args['best_record_inner']['Dice'] = pupil_dice
        if train_args['gpu_ids']:
            torch.save(net.module.state_dict(), os.path.join(checkpoint_path, 'for_inner.pth'))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, 'optimizer_inner.pt'))
            print("here, at pupil_e1 better than train_args")
            ##################### debugging pth write #########################
            # with open(checkpoint_path+"ss.text", "w") as f:
                # f.write("sss")
            ###################################################################
        else:
            torch.save(net.state_dict(), os.path.join(checkpoint_path, 'for_inner.pth'))
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, 'optimizer_inner.pt'))
        inner_checkpoints_name = 'epoch_%d_e1_%.7f_iou_%.7f_dice_%.7f' % (epoch, pupil_e1, pupil_iou, pupil_dice)
        logging.info(f'Saved pupil checkpoints {inner_checkpoints_name}.pth!')


    net.train()
    return val_loss


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    json_root = "./"
    args = get_args() #path to nucdre_images.zip
    print("debug => ", args.debug)
    # train_args = {
    #     'epoch_num': args.epoch_num,
    #     'batch_size': args.batch_size,
    #     'lr': args.lr,
    #     'checkpoints': args.checkpoints,  # empty string denotes learning from scratch
    #     'log_name': args.log_name,
    #     'print_freq': 20,
    #     'gpu_ids': args.gpu_ids
    # }
    train_args = {
        'epoch_num': args.epochs,
        'batch_size': args.batchsize,
        'lr': 0.002,
        'checkpoints': "",  # empty string denotes learning from scratch
        'log_name': "trial.log",
        'print_freq': 1000,
        'gpu_ids': None
    }
    print("args => ", train_args)

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    check_mkdir('experiments')
    log_path = os.path.join('experiments', experiment_name + '_' + start_time + '_' + train_args['log_name'].split('.')[0])
    check_mkdir(log_path)
    checkpoint_path = os.path.join(log_path, 'checkpoints')
    check_mkdir(checkpoint_path)
    logging.basicConfig(
        filename=os.path.join(log_path,train_args['log_name']),
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    images_filename = args.input0
    with zipfile.ZipFile(images_filename, "r") as zip_ref:
        zip_ref.extractall("./")

    json_filename = args.input1
    with zipfile.ZipFile(json_filename, "r") as zip_ref:
        zip_ref.extractall("./")

    print("succesfully opened zip files")
    main(train_args, json_root, args.debug) # creates logs under experiment/

    # args.output.mkdir()
    output_folder = args.output / "experiments"
    shutil.copytree("experiments/", output_folder)