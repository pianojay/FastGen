"""Evaluation child whose parent owns the cumulative budget clock."""
import argparse
import json
import torch
from soflow_fastgen.evaluation import evaluate_checkpoint

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--nfes',type=int,nargs='+',required=True);p.add_argument('--seeds',type=int,nargs='+',required=True)
    a=p.parse_args();c=json.load(open(a.config));torch.set_num_threads(c['torch_threads'])
    evaluate_checkpoint(a.checkpoint,c,a.nfes,a.seeds)
