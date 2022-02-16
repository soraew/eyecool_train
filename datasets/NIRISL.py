import os
import random
import json
import sys
from pathlib import Path
from sys import path
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


# root = '.../NIRISL-dataset/' # change this as you need
# root  = "../kci_eye"
root = ''

# it's data_root, but it's actually json_root
def make_dataset_list(data_root, mode):
    if mode == "train":
        # NIR-ISL2021-nucdre/jsons/dataset_f-1_train.jsonl
        jsonl_path = os.path.join(data_root, "jsons/dataset_f-1_train.jsonl")
        # jsonl_path = "jsons/dataset_f-1_train.jsonl" >> worked

    if mode == "test": ### changed
        # NIR-ISL2021-nucdre/jsons/dataset_f-1_evaluation.jsonl
        jsonl_path = os.path.join(data_root, "jsons/dataset_f-1_evaluation.jsonl")
        
    if mode == "valuation":
        # NIR-ISL2021-nucdre/jsons/dataset_f-1_validation.jsonl
        jsonl_path = os.path.join(data_root, "jsons/dataset_f-1_validation.jsonl")
        
    with open(jsonl_path, 'r') as json_file:
        json_list = list(json_file)

    data_list = []
    for json_str in json_list:
        result = json.loads(json_str)
        # added below code
        result = json.loads(json_str)
        is_there_iris = 0
        is_there_pupil = 0
        for region in result["regions"]:
            if region["label"] == "iris": is_there_iris = 1;
            elif region["label"] == "pupil": is_there_pupil = 1;
        enough = is_there_pupil * is_there_iris 
        if enough == 0:#iris mask, pupil maskどちらともないとlist に追加されない
            pass
        else:
            data_list.append(result)

    #ここでrandom shuffleしてみた  必要なのかは定かではない
    random.seed(0)
    random.shuffle(data_list)

    return data_list




images_dir = "nucdre_images_04/" # changed from images_dir = "../nucdre_images_04/" 


class eyeDataset(Dataset):
    '''
    args:
        dataset_name = "CASIA-Iris-Mobile-V1.0" # we are using nucdre as M1 size
        mode(str): 'train','test' == 'evaluation', 'valuation'
        # test は evaluation, validationどちらに対応するのか謎
        transform(dict): {'train': train_augment, 'test': test_augment}

    return(dict): {
        'image': aug_img,
        "iris_mask":aug_iris_mask,
        "pupil_mask":aug_pupil_mask
    }
    '''
    def __init__(self, dataset_name, data_root, mode, transform=None):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.mode = mode
        self.transform = transform 
        # again, here we don't have self.data_path
        self.data_list = make_dataset_list(data_root, mode)
        
    def __len__(self):
        return len(self.data_list)
    
    # idxが存在しないものもある
    def __getitem__(self, idx):
            
        ### ここよくわからん
        if self.mode == "test": ### changed from evalutation
            image_name = "-".join(self.data_list[idx]["image_path"].split("/"))
            # deleted image_name
            # [240, 240, 3]->[240, 240], 白黒へ
            image = Image.open(images_dir + self.data_list[idx]["image_path"]).convert("L")
            image = np.array(image)
            image = cv2.resize(image, (400, 400)) #### changed
            
            if self.transform is not None:
                image = np.asarray(image)
                aug_data = self.transform(image=image)
                aug_image = aug_data["image"]
                image = Image.fromarray(aug_image)
                
            image = transforms.ToTensor()(image)
            return {
                'image_name':image_name,
                'image':image
            }
        ###

        # this is same as the image path
        image_name = "-".join(self.data_list[idx]["image_path"].split("/"))
        # stopped converting to back and white
        image = Image.open(images_dir + self.data_list[idx]["image_path"])#.convert("L") #No such file or directory: '../nucdre_images_04/img12/StudyRHMI/Participant019/l/eyesstream_10009.jpg'
        # print("loaded!!!!!!!!!!")
        image = np.array(image)
        # print("image.shape>>>>>>>: ", image.shape)==>(240, 240)
        image = cv2.resize(image, (400, 400)) #### changed
        # print("new size >>>:", image.shape)==>(400, 400)
        regions = self.data_list[idx]["regions"]

        
        # not making this [240, 240, 3]for now
        iris_mask = np.zeros([240, 240], dtype = np.uint8)
        pupil_mask = np.zeros([240, 240], dtype = np.uint8)
        for region in regions:
            if region["label"] == "eyeball":
                pass
            elif region["label"] == "iris":
                points = np.array([region["position"]["points"]])
                cv2.fillPoly(iris_mask, points, 255)
                iris_mask = cv2.resize(iris_mask, (400, 400))
            elif region["label"] == "pupil":
                points = np.array([region["position"]["points"]])
                cv2.fillPoly(pupil_mask, points, 255)
                pupil_mask = cv2.resize(pupil_mask, (400, 400))
            else:
                print("region label >", region["label"])
                
        if self.transform is not None:
            image = np.asarray(image)
            iris_mask = np.asarray(iris_mask)
            pupil_mask = np.asarray(pupil_mask)
            # iris/pupil mask はそのまま
            mask_list = [iris_mask, pupil_mask]
            
            
            aug_data = self.transform(image=image, masks=mask_list)
            aug_image, aug_mask_list = aug_data['image'], aug_data['masks']
            
            image = Image.fromarray(aug_image)
            iris_mask = Image.fromarray(aug_mask_list[0])
            pupil_mask = Image.fromarray(aug_mask_list[1])
            
        aug_image = transforms.ToTensor()(image)
        aug_iris_mask = transforms.ToTensor()(iris_mask)
        aug_pupil_mask = transforms.ToTensor()(pupil_mask)
        
        return {
            "image_name":image_name,
            "image":aug_image,
            "iris_mask":aug_iris_mask,
            "pupil_mask":aug_pupil_mask
        }
                

if __name__ == "__main__":
    make_dataset_list(data_root="", mode="train")