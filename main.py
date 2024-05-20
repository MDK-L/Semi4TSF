import torch

import os
import numpy as np
from numpy import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.loss import NTXentLoss
from datetime import datetime
import argparse
from models.TC import TC
from utils import *
from models.model import base_Model
from dataloader.augmentations import DataTransform
from supervised_contrastive_loss import supervised_contrastive_loss
from data import get_loader
from data_provider.data_factory import data_provider
from models.model import base_Model
from models.TC import TC
from models.loss import NTXentLoss
start_time = datetime.now()


parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--label_ratio', type=float, default=0.1, help='labels ratio')
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--selected_dataset', default='ETTh2', type=str,
                    help='Dataset of choice: ETTh1, ETTh2')
parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--data', type=str, default='ETTh2', help='dataset type:ETTh1,ETTh2,ETTm1,weather')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
args = parser.parse_args()


device = init_dl_program(args.gpu)
experiment_description = args.experiment_description
data_type = args.selected_dataset
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

# Load Model
model = base_Model(configs, args).to(device)
temporal_contr_model = TC(configs, device).to(device)

predictor = nn.Linear(configs.features_len * configs.final_out_channels, args.pred_len).to(device)  

criterion = nn.MSELoss()
mae_loss = nn.L1Loss()

def rmse_loss(output, target):
    mse = criterion(output, target)
    rmse = torch.sqrt(mse)
    return rmse

def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


optimizer = torch.optim.AdamW(list(model.parameters()) + list(temporal_contr_model.parameters()) + list(predictor.parameters()), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0, last_epoch=-1)

contrastiveLoss = supervised_contrastive_loss(temperature=0.2, batch_size=args.batch_size, label_ratio=args.label_ratio)

fold = 5
loop=1

# t = time.time()
start = datetime.now()

for looptime in range(loop):
    dict_performance_fold = {}
    for curr_fold in range(fold):

        curr_fold = curr_fold + 1 

        dict_performance = {}

        train_labeled_loader, train_unlabeled_loader, test_loader = data_provider(args, configs)
        for epoch in range(1, configs.num_epoch + 1):

            model.train()
            temporal_contr_model.train()
            predictor.train()

            train_loss = []

            train_loss1 = []
            train_loss2 = []
            train_corrects = 0
            train_samples_count = 0

            for [x_labeled, y_labeled, aug_labeled, _, _], [x_unlabeled, _, aug, data_F, aug_F] in zip(train_labeled_loader, train_unlabeled_loader):
                optimizer.zero_grad()

                x_labeled = x_labeled.float().to(device)
                x_unlabeled = x_unlabeled.float().to(device)
                aug = aug.float().to(device)
                y_labeled = y_labeled.float().to(device)
                aug_labeled = aug_labeled.float().to(device)
                data_F, aug_F = data_F.float().to(device), aug_F.float().to(device)

                # _, features1 = model(x_unlabeled)
                # _, features2 = model(aug)

                # # normalize projection feature vectors
                # features1 = F.normalize(features1, dim=1)
                # features2 = F.normalize(features2, dim=1)



                # temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
                # temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)


                # # normalize projection feature vectors
                # zis = temp_cont_lstm_feat1
                # zjs = temp_cont_lstm_feat2

                # zis_shape=zis.size(0)

                # lambda1 = 0.5
                # lambda2 = 0.5
                # # lambda1 = 1
                # # lambda2 = 0.7
                # nt_xent_criterion = NTXentLoss(device, zis_shape, configs.Context_Cont.temperature,
                #                                configs.Context_Cont.use_cosine_similarity)
                # loss1 = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(zis, zjs) * lambda2

                predictions, features = model(x_unlabeled)
                predictions_aug, features_aug = model(aug) #对时域进行增强
    
                predictions_F, features_F= model(data_F)
                predictions_aug_F, features_aug_F = model(aug_F) #对频率进行增强
    
                # normalize projection feature vectors
                features = F.normalize(features, dim=1)#原时域序列
                features_aug = F.normalize(features_aug, dim=1)#原时域序列增强
    
                features_F= F.normalize(features_F, dim=1)#原序列的频率部分
                features_aug_F= F.normalize(features_aug_F, dim=1)#频率的增强
    
                temp_cont_loss1, _, temp_cont_lstm_feat1 = temporal_contr_model(features, features_aug)#原时间序列和时域增强的TC
                temp_cont_loss2, _, temp_cont_lstm_feat2 = temporal_contr_model(features_aug, features)
    
                _, freq_cont_loss1, freq_cont_lstm_feat1 = temporal_contr_model(features_F, features_aug_F)#原序列频率和频率增强的loss，这里面已经算过amp和phase的对比loss了
                _, freq_cont_loss2, freq_cont_lstm_feat2 = temporal_contr_model(features_aug_F, features_F)
    
                # normalize projection feature vectors
                zis_t = temp_cont_lstm_feat1# t,zis和zjs是c_t(s)和c_t(w)经过Non-Linear之后的数，然后再进行上下文对比
                zjs_t_aug = temp_cont_lstm_feat2# t_aug
    
                zks_f=freq_cont_lstm_feat1# f
                zls_f_aug=freq_cont_lstm_feat2# f_aug
                                                                                         
                lambda1 = 1
                lambda2 = 0.6
                lambda3 = 0.4
                nt_xent_criterion = NTXentLoss(device, configs.batch_size, configs.Context_Cont.temperature,
                                               configs.Context_Cont.use_cosine_similarity)#InfoNCE
                
                # loss_t=temp_cont_loss1 + temp_cont_loss2 #时域的上下文一致性
                loss_f=freq_cont_loss1 #时域的上下文一致性
                loss_tc=nt_xent_criterion(zis_t, zjs_t_aug) #时域一致性
                # loss_fc=nt_xent_criterion(zks_f, zls_f_aug) #频域一致性
                loss_tf=nt_xent_criterion(zis_t,zks_f)
    
                l_1, l_2, l_3 = nt_xent_criterion(zis_t, zls_f_aug), nt_xent_criterion(zjs_t_aug, zks_f), \
                            nt_xent_criterion(zjs_t_aug, zls_f_aug)
    
                loss_tfc=(1 + l_1 - loss_tf) + (1 + l_2 - loss_tf) + (1 + l_3 - loss_tf) # 时频一致性

                
                # loss1
                #loss1 = loss_t * lambda1+ (loss_tc + loss_fc) * lambda2 + loss_tfc * lambda3

                loss1 = loss_tc * lambda1+ loss_f * lambda2 + loss_tfc * lambda3
            

                '''supervised contrastive_loss + cross_entropy loss'''
                y_pred, features = model(x_labeled)
                y_aug_pred, features_aug = model(aug_labeled)
                
                y_pred = torch.cat([y_pred, y_aug_pred], dim=0)
                y_labeled = torch.cat([y_labeled, y_labeled], dim=0)
                loss2 = criterion(y_pred, y_labeled) # 这个其实是forecaster loss

                features = torch.cat([features, features_aug], dim=0)
                loss3 = contrastiveLoss(features.view(features.size()[0], -1), y_labeled) #supervised loss


                
                losstrain = 0.8* loss1 + loss2 + loss3
                torch.autograd.backward(losstrain)

                train_loss.append(losstrain.item())

                optimizer.step()

            print('Fold: {} \tTrain Epoch: {} \tLoss: {:.6f}, \tLoss1: {:.6f}, \tLoss2: {:.6f}, \tLoss3: {:.6f}'.format(curr_fold, epoch, losstrain.item(), loss1, loss2, loss3))


        test_loss = []
        mae=[]
        rmse=[]
        r2=[]

        model.eval()
        predictor.eval()
        
        with torch.no_grad():
            for x, y, _, _, _ in test_loader:
                x, y = x.float().to(device), y.float().to(device)

                predictions, features = model(x)
                mse_value = criterion(predictions, y)
                mae_value = mae_loss(predictions, y)
                rmse_value = rmse_loss(predictions, y)
                r2_value = r2_score(predictions, y)
        print(f'MSE: {mse_value.item()}, MAE: {mae_value.item()}, RMSE: {rmse_value.item()}, R^2: {r2_value.item()}')
                #test_loss.append(criterion(predictions, y).detach().cpu().numpy())
               # mae.append(mae_loss(predictions, y).detach().cpu().numpy())
                
       # print('MSE, MAE, :', np.mean(test_loss),np.mean(mae))

# 假设 output 和 target 已经定义







