import collections
import json
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import *
from utils.Trainer import *
from model.DOCTOR import *
import warnings
warnings.filterwarnings("ignore")

class Run():
    def __init__(self,config):
        self.dataset = config['dataset']
        self.mode = config['mode']
        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.early_stop = config['early_stop']
        self.device = config['device']
        self.lr = config['lr']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.path_ckp=config['path_ckp']
        self.path_tb=config['path_tb']
        self.inference_ckp=config['inference_ckp']
        self.dg=config['dg']
        self.diffusion=config['diffusion']

    def get_dataloader(self,data_path):
        dataset=FakingRecipe_Dataset(data_path,self.dataset)
        collate_fn=collate_fn_FakeingRecipe
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        return dataloader

    def main(self):
        self.model = FakingRecipe_Model(self.dataset)
        project_root = os.path.dirname(os.path.abspath(__file__))

        if self.mode=='train':
            if self.dataset=='fakesv':
                data_split_dir='./data/FakeSV/data-split/'
                save_predict_result_path='./predict_result/FakeSV/domain/'
            elif self.dataset=='fakett':
                data_split_dir='./data/FakeTT/data-split/'
                save_predict_result_path='./predict_result/FakeTT/domain/'
            
            # train_data_path=data_split_dir+'vid_time3_train.txt'
            # test_data_path=data_split_dir+'vid_time3_test.txt'
            # val_data_path=data_split_dir+'vid_time3_val.txt'
            if self.dataset=='fakesv':
                train_data_path=project_root + "/FakeSV_Domain_output/split/train_domain.txt"
                test_data_path=project_root + "/FakeSV_Domain_output/split/test_domain.txt"
                val_data_path=project_root + "/FakeSV_Domain_output/split/test_domain.txt"

                train_count = sum(1 for _ in open(project_root + "/FakeSV_Domain_output/split/train_domain.txt", 'r',
                                                  encoding='utf-8'))
                test_count = sum(1 for _ in open(project_root + "/FakeSV_Domain_output/split/test_domain.txt", 'r',
                                                 encoding='utf-8'))
                val_count = sum(1 for _ in open(project_root + "/FakeSV_Domain_output/split/test_domain.txt", 'r',
                                                encoding='utf-8'))
            if self.dataset=='fakett':
                train_data_path = project_root + "/FakeTT_Domain_output/split/train_domain.txt"
                test_data_path = project_root + "/FakeTT_Domain_output/split/test_domain.txt"
                val_data_path = project_root + "/FakeTT_Domain_output/split/test_domain.txt"

                train_count = sum(
                    1 for _ in open(project_root + "/FakeTT_Domain_output/split/train_domain.txt", 'r',
                                    encoding='utf-8'))
                test_count = sum(
                    1 for _ in open(project_root + "/FakeTT_Domain_output/split/test_domain.txt", 'r',
                                    encoding='utf-8'))
                val_count = sum(1 for _ in open(project_root + "/FakeTT_Domain_output/split/test_domain.txt", 'r',
                                                encoding='utf-8'))
            print(f"Train data count: {train_count}\nTest data count: {test_count}\nValidation data count: {val_count}")

            data_load_time_start = time.time()
            train_dataloader=self.get_dataloader(train_data_path)
            test_dataloader=self.get_dataloader(test_data_path)
            val_dataloader=self.get_dataloader(val_data_path)
            dataloaders=dict(zip(['train','test','val'],[train_dataloader,test_dataloader,val_dataloader]))
            print ('data load time: %.2f' % (time.time() - data_load_time_start))
            trainer=Trainer(model=self.model,device=self.device,lr=self.lr,dataloaders=dataloaders,epoches=self.epoches,model_name='FakingRecipe',save_predict_result_path=save_predict_result_path,beta_c=self.alpha,beta_n=self.beta,beta_dg=self.gamma,early_stop=self.early_stop,save_param_path=self.path_ckp+self.dataset+"/",writer=SummaryWriter(self.path_tb+self.dataset+"/"),dg=self.dg,diffusion=self.diffusion)
            ckp_path=trainer.train()
            result=trainer.test(ckp_path)
        elif self.mode=='inference_test':
            if self.dataset=='fakesv':
                test_file='./data/FakeSV/data-split/vid_time3_test.txt'
                save_predict_result_path='./predict_result/FakeSV/'
            elif self.dataset=='fakett':
                test_file='./data/FakeTT/data-split/vid_time3_test.txt'
                save_predict_result_path='./predict_result/FakeTT/'
            dataloader=self.get_dataloader(test_file)
            inferncer=Inferencer(model=self.model,device=self.device,model_name='FakingRecipe',dataset=self.dataset,dataloader=dataloader,save_predict_result_path=save_predict_result_path)
            result=inferncer.inference(self.inference_ckp)
