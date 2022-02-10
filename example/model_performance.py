import os
import argparse
from numpy.lib.type_check import imag
import pandas as pd
import numpy as np
from thop.profile import profile
import tqdm
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import albumentations as A

from os import path
from pathlib import Path
import sys
path_root = Path(__file__).parents[1] #adding NIR-ISL2021 as root
sys.path.append(str(path_root))

import sys
# sys.path.append('.../NIR-ISL2021master/') # change as you need
from datasets.NIRISL import eyeDataset
from models.efficient_unet import EfficientUNet
from evaluation.eval_loc import evaluate_loc
# from location import get_edge #

# I add torchsummary for vis

# may want to change this to 1? this was originally 2
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cudnn.benchmark = True

# nucdre data is set to the same as M1, so we will be using M1 weights and dataset_name
# def get_args():
    # parser = argparse.ArgumentParser(description='Test paprmeters', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--dataset', default="CASIA-Iris-Mobile-V1.0", type=str, required=True, dest='dataset_name')
    # parser.add_argument('--ckpath', default="M1-checkpoints", type=str, required=True, dest='checkpoints_path')
    # return parser.parse_args()


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

# added save_dir to test() and main()
def main(test_args, save_dir):

    # assert test_args['dataset_name'] in ['CASIA-Iris-Africa','CASIA-distance', 'Occlusion', 'Off_angle', 'CASIA-Iris-Mobile-V1.0']
    ############################################# define a CNN #################################################
    net = EfficientUNet(num_classes=3) # deleted .cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])

    ########################################### dataset #############################################
    # M1 に合わせて(400x400の画像にしたので、checkpoints はM1のものを使う)
    test_augment = A.Compose([
        # A.Resize(320, 544) # for Africa dataset
    ])
    # wondering what to do with test_args
    # we are using mode "train" for calculation of scores
    test_dataset = eyeDataset(test_args['dataset_name'], data_root="", mode='train', transform=test_augment)
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False)
    print('The dataset {} is ready!'.format(test_args['dataset_name']))

    ######################################### test ############################################
    test(test_loader, net, test_args, save_dir)# added test args here


def test(test_loader, net, test_args, save_dir):# added test args here 
    print('start test......')


    # moved from __main__.py
    iris_edge_mask_raw_save_dir = os.path.join(save_dir, 'iris_edge_mask_raw') # for pred iris mask 
    check_mkdir(iris_edge_mask_raw_save_dir)

    pupil_edge_mask_raw_save_dir = os.path.join(save_dir, 'pupil_edge_mask_raw') # for pred pupil mask
    check_mkdir(pupil_edge_mask_raw_save_dir)

    # names, iris_circles_params, pupil_circles_params = [], [], []
    # state_dict = torch.load(os.path.join(test_args['checkpoints_path'], 'for_mask.pth'))
    # state_dict["module.heatmap4.loc.0.weight"] = state_dict.pop('module.loc4.loc.0.weight')
    # state_dict["module.heatmap3.loc.0.weight"] = state_dict.pop('module.loc3.loc.0.weight')
    # state_dict["module.heatmap2.loc.0.weight"] = state_dict.pop('module.loc2.loc.0.weight')
    # state_dict["module.heatmap4.loc.0.bias"] = state_dict.pop('module.loc4.loc.0.bias')
    # state_dict["module.heatmap3.loc.0.bias"] = state_dict.pop('module.loc3.loc.0.bias')
    # state_dict["module.heatmap2.loc.0.bias"] = state_dict.pop('module.loc2.loc.0.bias')
    # net.load_state_dict(state_dict)
    # net.eval()
    # for i, data in enumerate(test_loader):
    #     image_name, image = data['image_name'][0], data['image'] #BCHW
    #     print('testing the {}-th image: {}'.format(i+1, image_name))
    #     image = Variable(image).cuda()

    #     with torch.no_grad():# 推論！！
    #         outputs = net(image)

    #     pred_iris_mask, pred_pupil_mask =outputs['pred_iris_mask'], outputs['pred_pupil_mask']
    #     # pred_mask_pil = transforms.ToPILImage()((pred_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
    #     # pred_mask_pil.save(os.path.join(SegmentationClass_save_dir, image_name+'.png'))


    state_dict = torch.load(os.path.join(test_args['checkpoints_path'], 'for_inner.pth'), map_location=torch.device('cpu'))
    state_dict["module.heatmap4.loc.0.weight"] = state_dict.pop('module.loc4.loc.0.weight')
    state_dict["module.heatmap3.loc.0.weight"] = state_dict.pop('module.loc3.loc.0.weight')
    state_dict["module.heatmap2.loc.0.weight"] = state_dict.pop('module.loc2.loc.0.weight')
    state_dict["module.heatmap4.loc.0.bias"] = state_dict.pop('module.loc4.loc.0.bias')
    state_dict["module.heatmap3.loc.0.bias"] = state_dict.pop('module.loc3.loc.0.bias')
    state_dict["module.heatmap2.loc.0.bias"] = state_dict.pop('module.loc2.loc.0.bias')
    net.load_state_dict(state_dict)
    net.eval()

    inner_iris = {'E1':[], 'IoU':[], 'Dice':[], 'TP':[], 'FP':[], 'FN':[], 'recall':[], 'precision':[], 'F1':[]}
    inner_pupil = {'E1':[], 'IoU':[], 'Dice':[],  'TP':[], 'FP':[], 'FN':[], 'recall':[], 'precision':[], 'F1':[]}
    count = 0
    break_count = 20
    for i, data in tqdm.tqdm(enumerate(test_loader)):
        image_name, image = data['image_name'][0], data['image'] #BCHW
        gt_iris_mask, gt_pupil_mask = data['iris_mask'], data['pupil_mask']
        print('testing the {}-th image: {}'.format(i+1, image_name))
        image = Variable(image) #del cuda

        with torch.no_grad():
            outputs = net(image)

        pred_iris_mask, pred_pupil_mask = \
            outputs['pred_iris_mask'], outputs['pred_pupil_mask']
        # pred_iris_circle_mask, pred_iris_edge, iris_circles_param = get_edge(pred_iris_mask)

        #############################################################################
        ############### SHOW IMAGE OF PREDS AND GT MASKS ############################
        #############################################################################
        print("TYPE>>>>>>", type(pred_iris_mask))
        print("SHAPE>>>>>", pred_iris_mask.shape)
        print("IMG shape >>>>>>", image.shape)
        
        # plot mask and image
        # plt.imshow(image.numpy().reshape(400, 400, 3), "gray")
        # plt.imshow(pred)
        fig = plt.figure()
        # plt.imshow(gt_iris_mask.numpy().reshape(400,400,1),"Blues")
        pred_iris_mask = pred_iris_mask>0 * 1.
        pred_pupil_mask = pred_pupil_mask>0 * 1.
        # plt.imshow(image.reshape(400, 400, 1))
        image = image.numpy()
        image_show = np.mean(image, axis=1)
        image_show = image_show.reshape(400, 400, 1)
        plt.imshow(image_show)

        plt.imshow(pred_iris_mask.numpy().reshape(400,400,1), "magma", alpha=0.3)
        plt.show()
        fig = plt.figure()
        plt.imshow(image_show)
        plt.imshow(pred_pupil_mask.numpy().reshape(400,400,1), "magma", alpha=0.3)
        plt.show()
        #############################################################################
        #############################################################################
        #############################################################################

        # scoring output
        iris_dict = evaluate_loc(pred_iris_mask, gt_iris_mask)
        inner_iris["E1"].append(iris_dict["E1"])
        inner_iris["IoU"].append(iris_dict["IoU"])
        inner_iris["Dice"].append(iris_dict["Dice"])
        inner_iris["TP"].append(iris_dict["TP"])
        inner_iris["FP"].append(iris_dict["FP"])
        inner_iris["FN"].append(iris_dict["FN"])
        inner_iris["recall"].append(iris_dict["recall"])
        inner_iris["precision"].append(iris_dict["precision"])
        inner_iris["F1"].append(iris_dict["F1"])


        pupil_dict = evaluate_loc(pred_pupil_mask, gt_pupil_mask)
        inner_pupil["E1"].append(pupil_dict["E1"])
        inner_pupil["IoU"].append(pupil_dict["IoU"])
        inner_pupil["Dice"].append(pupil_dict["Dice"])
        inner_pupil["TP"].append(pupil_dict["TP"])
        inner_pupil["FP"].append(pupil_dict["FP"])
        inner_pupil["FN"].append(pupil_dict["FN"])
        inner_pupil["recall"].append(pupil_dict["recall"])
        inner_pupil["precision"].append(pupil_dict["precision"])
        inner_pupil["F1"].append(pupil_dict["F1"])



        # for division of scores later
        count += 1

        # saving pred_iris_mask files in iris_edge_mask_raw_save_dir
        pred_iris_mask_pil = transforms.ToPILImage()((pred_iris_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        # pred_iris_circle_mask_pil = transforms.ToPILImage()((pred_iris_circle_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        # pred_iris_edge_pil = transforms.ToPILImage()((pred_iris_edge[0]>0).to(dtype=torch.uint8)*255).convert('L')
        # iris_circles_params.append(iris_circles_param.cpu().numpy()[0])

        pred_iris_mask_pil.save(os.path.join(iris_edge_mask_raw_save_dir, image_name+'.png'))
        # pred_iris_circle_mask_pil.save(os.path.join(iris_edge_mask_save_dir, image_name+'.png'))
        # pred_iris_edge_pil.save(os.path.join(Outer_Boundary_save_dir, image_name+'.png'))
        if count > break_count:
            break
    
    for key in inner_iris.keys():
        # print(f"inner_iris[{key}]>>", inner_iris[key])
        inner_iris[key] = np.array(inner_iris[key]).mean()
        # print(f"inner_pupil[{key}]>>", inner_pupil[key])
        inner_pupil[key] = np.array(inner_pupil[key]).mean()
    

    state_dict = torch.load(os.path.join(test_args['checkpoints_path'], 'for_outer.pth') ,map_location=torch.device('cpu'))
    state_dict["module.heatmap4.loc.0.weight"] = state_dict.pop('module.loc4.loc.0.weight')
    state_dict["module.heatmap3.loc.0.weight"] = state_dict.pop('module.loc3.loc.0.weight')
    state_dict["module.heatmap2.loc.0.weight"] = state_dict.pop('module.loc2.loc.0.weight')
    state_dict["module.heatmap4.loc.0.bias"] = state_dict.pop('module.loc4.loc.0.bias')
    state_dict["module.heatmap3.loc.0.bias"] = state_dict.pop('module.loc3.loc.0.bias')
    state_dict["module.heatmap2.loc.0.bias"] = state_dict.pop('module.loc2.loc.0.bias')
    net.load_state_dict(state_dict)
    net.eval()

    outer_iris = {'E1':[], 'IoU':[], 'Dice':[], 'TP':[], 'FP':[], 'FN':[], 'recall':[], 'precision':[], 'F1':[]}
    outer_pupil = {'E1':[], 'IoU':[], 'Dice':[], 'TP':[], 'FP':[], 'FN':[], 'recall':[], 'precision':[], 'F1':[]}
    count = 0 # reset count to zero
    for i, data in enumerate(test_loader):

        image_name, image = data['image_name'][0], data['image'] #BCHW
        gt_iris_mask, gt_pupil_mask = data['iris_mask'], data['pupil_mask']
        print('testing the {}-th image: {}'.format(i+1, image_name))
        image = Variable(image)# del cuda

        with torch.no_grad():
            outputs = net(image)

        pred_iris_mask, pred_pupil_mask = \
            outputs['pred_iris_mask'], outputs['pred_pupil_mask']
        # post processing
        # pred_pupil_circle_mask, pred_pupil_egde, pupil_circles_param = get_edge(pred_pupil_mask)
        
        iris_dict = evaluate_loc(pred_iris_mask, gt_iris_mask)
        outer_iris["E1"].append(iris_dict["E1"])
        outer_iris["IoU"].append(iris_dict["IoU"])
        outer_iris["Dice"].append(iris_dict["Dice"])
        outer_iris["TP"].append(iris_dict["TP"])
        outer_iris["FP"].append(iris_dict["FP"])
        outer_iris["FN"].append(iris_dict["FN"])
        outer_iris["recall"].append(iris_dict["recall"])
        outer_iris["precision"].append(iris_dict["precision"])
        outer_iris["F1"].append(iris_dict["F1"])

        pupil_dict = evaluate_loc(pred_pupil_mask, gt_pupil_mask)
        outer_pupil["E1"].append(pupil_dict["E1"])
        outer_pupil["IoU"].append(pupil_dict["IoU"])
        outer_pupil["Dice"].append(pupil_dict["Dice"])
        outer_pupil["TP"].append(pupil_dict["TP"])
        outer_pupil["FP"].append(pupil_dict["FP"])
        outer_pupil["FN"].append(pupil_dict["FN"])
        outer_pupil["recall"].append(pupil_dict["recall"])
        outer_pupil["precision"].append(pupil_dict["precision"])
        outer_pupil["F1"].append(pupil_dict["F1"])
        # names.append(image_name)
        # pupil_circles_params.append(pupil_circles_param.cpu().numpy().tolist()[0])

        pred_pupil_mask_pil = transforms.ToPILImage()((pred_pupil_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        # pred_pupil_circle_mask_pil = transforms.ToPILImage()((pred_pupil_circle_mask[0]>0).to(dtype=torch.uint8)*255).convert('L')
        # pred_pupil_egde_pil = transforms.ToPILImage()((pred_pupil_egde[0]>0).to(dtype=torch.uint8)*255).convert('L')

        # Saving pupil mask
        pred_pupil_mask_pil.save(os.path.join(pupil_edge_mask_raw_save_dir, image_name+'.png'))
        # pred_pupil_circle_mask_pil.save(os.path.join(pupil_edge_mask_save_dir, image_name+'.png'))
        # pred_pupil_egde_pil.save(os.path.join(Inner_Boundary_save_dir, image_name+'.png'))
        count += 1
        if count > break_count:
            break

    for key in outer_iris.keys():
        # using previous count variable for division
        outer_iris[key] = np.array(outer_iris[key]).mean()
        outer_pupil[key] = np.array(outer_pupil[key]).mean()

    ############################### ? not really sure about here ? ###############################
    # iris_circles_params = np.asarray(iris_circles_params)
    # pupil_circles_params =np.asarray(pupil_circles_params)
    # params_path = save_dir + '/test_params.xlsx'
    # params_data = pd.DataFrame({
    #     'name':names,
    #     'ix':iris_circles_params[:,0],
    #     'iy':iris_circles_params[:,1],
    #     'ih':iris_circles_params[:,2],
    #     'iw':iris_circles_params[:,3],
    #     'ir':iris_circles_params[:,4],
    #     'px':pupil_circles_params[:,0],
    #     'py':pupil_circles_params[:,1],
    #     'ph':pupil_circles_params[:,2],
    #     'pw':pupil_circles_params[:,3],
    #     'pr':pupil_circles_params[:,4]
    #     })
    # params_data.to_excel(params_path)

    print('test done!')
    print("iterations >> ", count)
    print("inner_iris >> ", inner_iris)
    print("inner_pupil >> ", inner_pupil)
    print("outer_iris >> ", outer_iris)
    print("outer_pupil >> ", outer_pupil)


    net.train()

# using this in __main__.py at root
if __name__ == '__main__':
    # args = get_args()
    # test_args = {
    #     'dataset_name': args.dataset_name,
    #     'checkpoints_path': args.checkpoints_path
    # }
    test_args = {
        #   'dataset_name': args.dataset_name,
        'dataset_name' : 'CASIA-Iris-Mobile-V1.0',
        #   'checkpoints_path': args.checkpoints_path
        'checkpoints_path' : 'M1-checkpoints'
        }
    
    check_mkdir('./test-result')
    save_dir = os.path.join('test-result', test_args['dataset_name'])
    check_mkdir(save_dir)



    # SegmentationClass_save_dir = os.path.join(save_dir, 'SegmentationClass') # for pred mask
    # check_mkdir(SegmentationClass_save_dir)
    # Inner_Boundary_save_dir = os.path.join(save_dir, 'Inner_Boundary') # for pred pupil edge
    # check_mkdir(Inner_Boundary_save_dir)
    # Outer_Boundary_save_dir = os.path.join(save_dir, 'Outer_Boundary')# for pred iris edge
    # check_mkdir(Outer_Boundary_save_dir)
    iris_edge_mask_raw_save_dir = os.path.join(save_dir, 'iris_edge_mask_raw') # for pred iris mask 
    check_mkdir(iris_edge_mask_raw_save_dir)
    # iris_edge_mask_save_dir = os.path.join(save_dir, 'iris_edge_mask')
    # check_mkdir(iris_edge_mask_save_dir)
    pupil_edge_mask_raw_save_dir = os.path.join(save_dir, 'pupil_edge_mask_raw') # for pred pupil mask
    check_mkdir(pupil_edge_mask_raw_save_dir)
    # pupil_edge_mask_save_dir = os.path.join(save_dir, 'pupil_edge_mask')
    # check_mkdir(pupil_edge_mask_save_dir)

    main(test_args)
