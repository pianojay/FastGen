import argparse
import json
import torch
from soflow_fastgen.common import Budget
from soflow_fastgen.evaluation import evaluate_checkpoint

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/pokemon.json')
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--nfes',type=int,nargs='+',default=[1,2,3,4])
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    args=parser.parse_args()
    config=json.load(open(args.config))
    torch.set_num_threads(config['torch_threads'])
    budget=Budget(config['run_dir']+'/budget.json',config['budget_seconds'])
    try:
        evaluate_checkpoint(args.checkpoint,config,args.nfes,args.seeds,budget)
    finally:
        budget.save()
