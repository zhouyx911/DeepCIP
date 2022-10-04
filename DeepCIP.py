import os
import shutil
import warnings
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


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', default=None, help='', type=str)
parser.add_argument('--input_file', default=None, help='', type=str)
parser.add_argument('--outfile', default='./results/predict.out', help='', type=str)
parser.add_argument('--bs', default=32, help='batch_size', type=int)
parser.add_argument('--c', default=0.5, help='predictive threshold', type=float)
parser.add_argument('--mode', default=0, help='select predicted mode', type=int)
parser.add_argument('--w', default=174, help='window_size', type=int)
parser.add_argument('--s', default=50, help='step', type=int)
args = parser.parse_args()

warnings.filterwarnings('ignore')


pred_struc_dir = './pred_struc/'
if not os.path.exists(pred_struc_dir):
    os.mkdir(pred_struc_dir)
else:
    print(pred_struc_dir + ' exist')
    
structure_pred = subprocess.call(['RNAplfold -W 150 -c 0.001 --noLP --auto-id --id-digits 5 <' + f'{args.input_file}'], shell=True, stdout=subprocess.PIPE)

for struc_file in os.listdir('./'):
    if struc_file.endswith('_dp.ps'):
        shutil.move(struc_file, pred_struc_dir + struc_file)


#get_fasta_seq_name
#get_seq
name_ls_m0 = []
seq_ls_m0 = []
with open(args.input_file, 'r') as f1:
    fa_data = f1.read().splitlines()
    for line in fa_data:
        if '>' in line:
            name_ls_m0.append(line[1:])
        else:
            line.replace('U', 'T')
            seq_ls_m0.append(line)

name_ls_m1 = []
seq_ls_m1 = []
if args.mode == 1:
    for k in range(len(seq_ls_m0)):
        subseq, subname = mode1_process(seq_ls_m0[k], name_ls_m0[k], window_size=args.w, step=args.s)
        seq_ls_m1.extend(subseq)
        name_ls_m1.extend(subname)
else:
    pass


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

train_config = config
max_epochs = 2000
gpu = 0

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

data_test = RNAGraphDataset(dataset=args.data_name, rna_graph=pred_struc_dir,\
     xr=(seq_ls_m0 if args.mode == 0 else seq_ls_m1), \
     xs=(np.array(seq_ls_m0) if args.mode == 0 else np.array(seq_ls_m1)))

ckpt_paths1 = './model/Fusion/set1/version_0/checkpoints/epoch=5-step=168.ckpt'
ckpt_paths2 = './model/Fusion/set2/version_0/checkpoints/epoch=6-step=196.ckpt'
ckpt_paths3 = './model/Fusion/set3/version_1/checkpoints/epoch=7-step=224.ckpt'

# test_labels = torch.FloatTensor(data_test.data.y)

loader = DataLoader(dataset=data_test, batch_size=args.bs)


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
    if i < args.c:
        ires_label = 'Non-circires'
    else:
        ires_label = 'Circires'
    ires_labels.append(ires_label)

if args.mode == 0:
    with open(args.outfile, 'w') as fout:
        for n in range(len(ensemble_probs)):
            print(name_ls_m0[n],ensemble_probs[n],ires_labels[n],file=fout)
if args.mode == 1:
    with open(args.outfile, 'w') as fout:
        for n in range(len(ensemble_probs)):
            print(name_ls_m1[n],ensemble_probs[n],ires_labels[n],file=fout)

shutil.rmtree(pred_struc_dir)
for struc_file in os.listdir('./'):
    if struc_file.endswith('_dp.ps'):
        os.remove(struc_file)

print('Prediction complete!')