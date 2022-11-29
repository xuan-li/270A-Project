import mmcv
import argparse
import torch
from lib import Timer

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    return parser

if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    problem = mmcv.Config.fromfile(args.config)['problem']
    method = problem["method"]
    if torch.cuda.is_available():
        device = torch.device(problem["device"])
    else:
        device = torch.device('cpu')
    with Timer("recon"):
        X_recon = method(problem["observed_matrix"].to(device), 
                        problem["mask"].to(device), 
                        problem["step"], 
                        problem["rank"], 
                        100000, 
                        problem["tol"])
    error = torch.abs(problem["gt"] - X_recon.cpu()).max()
    print(f"Error: {error.item()}")

