import os
import sys
from numpy.lib.npyio import save
import torch
from torch.autograd import Variable

from example.model_performance import check_mkdir, main#, get_args
from datasets.NIRISL import make_dataset_list

if __name__ == '__main__':
   # print(make_dataset_list(data_root="", mode="train")[:10]) >> ok
   #  args = get_args()
   test_args = {
   #   'dataset_name': args.dataset_name,
      'dataset_name' : 'CASIA-Iris-Mobile-V1.0',
   #   'checkpoints_path': args.checkpoints_path
      'checkpoints_path' : "example/submission_1-checkpoints/M1-checkpoints" # 'M1-checkpoints' 
   }

   check_mkdir('./test-result')
   save_dir = os.path.join('test-result', test_args['dataset_name'])
   check_mkdir(save_dir)
   
   main(test_args, save_dir)
   
 
   


   # iris_edge_mask_raw_save_dir = os.path.join(save_dir, 'iris_edge_mask_raw') # for pred iris mask 
   # check_mkdir(iris_edge_mask_raw_save_dir)

   # pupil_edge_mask_raw_save_dir = os.path.join(save_dir, 'pupil_edge_mask_raw') # for pred pupil mask
   # check_mkdir(pupil_edge_mask_raw_save_dir)


   # main(test_args)
