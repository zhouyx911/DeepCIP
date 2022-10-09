import os
import shutil
import warnings
import datetime
import torch
import torch.nn as nn
import argparse
import subprocess
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer, seed_everything
from torch_geometric.loader import DataLoader
from compile.fusion_gnn import PL_Fusion
from compile.create_graph_predict import RNAGraphDataset
from compile.mode import mode1_process



def DeepCIP_predict():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--data_name', default=None, help='The name of your input dataset.', metavar='', type=str)
    parser.add_argument('-i','--input_file', default=None, help='Input file for prediction. (*.fasta or *.fa file)', metavar='', type=str)
    parser.add_argument('-b','--batch_size', default=32, help='Batch size. (default=32)  "--bs 32" means every 32 sampels constitute a prediction batch. This parameter affects the speed and       results of the prediction. The larger the batch size, the faster the prediction, as far as your machine allows. (If the lengths of your input sequences vary greatly, it is recommended that you do not use a large batch size, or you can put sequences of similar lengths together for prediction)', metavar='', type=int)
    parser.add_argument('-c','--cut_off', default=0.5, help='Prediction threshold. (default=0.5)', metavar='', type=float)
    parser.add_argument('-m','--mode', default=0, choices=[0, 1], help='The mode of prediction. (default=0)  mode 0: Prediction directly on the input sequence.  mode 1: The input sequence is partitioned by length w and interval s, and then the partitioned sequence is predicted. (w and s can be set by --w and --s, respectively)', metavar='', type=int)
    parser.add_argument('-w','--window_size', default=174, help='window size (default=174). See --mode description. It can be ignore when mode=0.', metavar='', type=int)
    parser.add_argument('-s','--step', default=50, help='step (default=50). See --mode description. It can be ignore when mode=0.', metavar='', type=int)
    args = parser.parse_args()

    warnings.filterwarnings('ignore')

    start = datetime.datetime.now()
    print(f'{start}: Prediction starting... \n')
    print('------------------------------------------------------------------')


    pred_struc_dir = './pred_struc/'
    if not os.path.exists(pred_struc_dir):
        os.mkdir(pred_struc_dir)
    else:
        print(pred_struc_dir + ' exist')
        


    #get_fasta_seq_name
    #get_seq
    name_ls_m0 = []
    seq_ls_m0 = []
    with open('./data/' + f'{args.input_file}', 'r') as f1:
        fa_data = f1.read().splitlines()
        for line in fa_data:
            if '>' in line:
                name_ls_m0.append(line[1:])
            else:
                line.replace('U', 'T')
                seq_ls_m0.append(line)

    print('Structure prediction...')

    if args.mode == 0:
        structure_pred0 = subprocess.call(['RNAplfold -W 150 -c 0.001 --noLP --auto-id --id-digits 5 <' + f'./data/{args.input_file}'], shell=True, stdout=subprocess.PIPE)

    else:
        name_ls_m1 = []
        seq_ls_m1 = []
        for k in range(len(seq_ls_m0)):
            subseq, subname = mode1_process(seq_ls_m0[k], name_ls_m0[k], window_size=args.window_size, step=args.step)
            seq_ls_m1.extend(subseq)
            name_ls_m1.extend(subname)
        with open(f'./data/process_{args.input_file}','w') as subfile:
            for l in range(len(seq_ls_m1)):
                subfile.write('>' + name_ls_m1[l] + '\n' + seq_ls_m1[l] + '\n')

        structure_pred1 = subprocess.call(['RNAplfold -W 150 -c 0.001 --noLP --auto-id --id-digits 5 <' + f'./data/process_{args.input_file}'], shell=True, stdout=subprocess.PIPE)

    for struc_file in os.listdir('./'):
        if struc_file.endswith('_dp.ps'):
            shutil.move(struc_file, pred_struc_dir + struc_file)

    seed = 37
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

    config = {
            'num_features': 4,
            'gcn_drop': 0.2,
            'graph_pool': 'sum',
            'output_dim': 1,
            'learning_rate': 8e-5,
            'batch_size': 256,
            'optim': 'Adam',
            'input_size': 4,
            'hidden_size': 64,
            'window_size': 3,
            'steps': 9,
            'sentence_nodes': 1,
            'slstm_drop': 0.0,
            'batch_first': False,
            'acti_func': 'ReLU',
            'seq_weight': 0.5,
            'struc_weight': 0.5
            
        }


    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    data_test = RNAGraphDataset(dataset=args.data_name, rna_graph=pred_struc_dir,\
        xr=(seq_ls_m0 if args.mode == 0 else seq_ls_m1), \
        xs=(np.array(seq_ls_m0) if args.mode == 0 else np.array(seq_ls_m1)))

    loader = DataLoader(dataset=data_test, batch_size=args.batch_size)

    ckpt_paths1 = './model/Fusion/set1/version_0/checkpoints/epoch=5-step=168.ckpt'
    ckpt_paths2 = './model/Fusion/set2/version_0/checkpoints/epoch=6-step=196.ckpt'
    ckpt_paths3 = './model/Fusion/set3/version_1/checkpoints/epoch=7-step=224.ckpt'



    preds_all = []

    preds1 = np.empty(0)
    seq_ckpt_paths1 = './model/S_LSTM/set1/version_1/checkpoints/epoch=12-step=364.ckpt'
    struc_ckpt_paths1 = './model/GCN/set1/version_0/checkpoints/epoch=93-step=2632.ckpt'
    model1 = PL_Fusion.load_from_checkpoint(ckpt_paths1, config=config, seq_ckpt_path=seq_ckpt_paths1, struc_ckpt_path=struc_ckpt_paths1, map_location=device)
    model1.eval()

    preds2 = np.empty(0)
    seq_ckpt_paths2 = './model/S_LSTM/set2/version_1/checkpoints/epoch=1-step=56.ckpt'
    struc_ckpt_paths2 = './model/GCN/set2/version_0/checkpoints/epoch=42-step=1204.ckpt'
    model2 = PL_Fusion.load_from_checkpoint(ckpt_paths2, config=config, seq_ckpt_path=seq_ckpt_paths2, struc_ckpt_path=struc_ckpt_paths2, map_location=device)
    model2.eval()

    preds3 = np.empty(0)
    seq_ckpt_paths3 = './model/S_LSTM/set3/version_1/checkpoints/epoch=15-step=448.ckpt'
    struc_ckpt_paths3 = './model/GCN/set3/version_0/checkpoints/epoch=6-step=196.ckpt'
    model3 = PL_Fusion.load_from_checkpoint(ckpt_paths3, config=config, seq_ckpt_path=seq_ckpt_paths3, struc_ckpt_path=struc_ckpt_paths3, map_location=device)
    model3.eval()

    with torch.no_grad():
        for test_data in loader:
            test_data = test_data.to(device)
            pred1 = model1(test_data).squeeze(-1)
            preds1 = np.append(preds1, pred1.cpu().numpy())
            pred2 = model2(test_data).squeeze(-1)
            preds2 = np.append(preds2, pred2.cpu().numpy())
            pred3 = model3(test_data).squeeze(-1)
            preds3 = np.append(preds3, pred3.cpu().numpy())


    preds_all.append(preds1)
    preds_all.append(preds2)
    preds_all.append(preds3)
    ensemble_probs = np.mean(preds_all, axis=0)

    ires_labels = []
    for i in ensemble_probs:
        if i < args.cut_off:
            ires_label = 'Non-circires'
        else:
            ires_label = 'Circires'
        ires_labels.append(ires_label)

    out_dir = './results'
    out_name = args.input_file.split('/')[-1].split('.')[0]
    outfile = f'{out_dir}/{out_name}_mode_{args.mode}.result'

    if args.mode == 0:
        with open(outfile, 'w') as fout:
            for n in range(len(ensemble_probs)):
                print(name_ls_m0[n],ensemble_probs[n],ires_labels[n],file=fout)
    if args.mode == 1:
        with open(outfile, 'w') as fout:
            for n in range(len(ensemble_probs)):
                print(name_ls_m1[n],ensemble_probs[n],ires_labels[n],file=fout)

    shutil.rmtree(pred_struc_dir)
    for struc_file in os.listdir('./'):
        if struc_file.endswith('_dp.ps'):
            os.remove(struc_file)

    print('------------------------------------------------------------------')
    end = datetime.datetime.now()
    print(f'{end}: Prediction complete! \n')

if __name__ == "__main__":
    DeepCIP_predict()