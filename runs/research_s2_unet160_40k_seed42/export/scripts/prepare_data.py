import argparse
import json
from soflow_fastgen.data import prepare

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/pokemon.json')
    args = parser.parse_args()
    config = json.load(open(args.config))
    result = prepare(config['data_root'], config['partition'], config['seed'])
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('optimization_indices','development_indices','class_counts','exact_duplicate_groups')}, indent=2))
    print('Exact duplicate groups:', len(result['exact_duplicate_groups']))
