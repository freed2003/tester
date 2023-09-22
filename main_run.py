import argparse
import numpy as np
from model import model
from data_loader import data_loader,synthetic_data
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization
def main(parameters):
    orig_data = []
    if args.synthetic:
        orig_data = synthetic_data(10000, 100)
    else:
        orig_data = data_loader(100)
    print("Generating data")
    new_data = model(orig_data)

    # metrics = {}
    # ds = []
    # for _ in range(10):
    #     temp = discriminative_score_metrics(orig_data, new_data)
    #     ds.append(temp)
    # metrics['discriminative'] = np.mean(ds)

    # pred = []
    # for __ in range(10):
    #     temp = predictive_score_metrics(orig_data, new_data)
    #     pred.append(temp)

    # metrics['predictive'] = np.mean(pred)

    visualization(orig_data, new_data, 'pca')
    visualization(orig_data, new_data, 'tsne')

    # print(metrics)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--synthetic',
        help='use python generated data instead',
        action='store_true'
    )
    args = parser.parse_args() 
    main(args)