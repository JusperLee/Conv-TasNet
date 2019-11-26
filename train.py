import sys
sys.path.append('./options')
import argparse
from utils import get_logger
from option import parse
from DataLoaders import make_dataloader
from Conv_TasNet import ConvTasNet
def main():
    ### Reading option
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',type=str,help='Path to option YAML file.')
    args = parser.parse_args()
    opt = parse(args.opt,is_tain=True)
    net = ConvTasNet(**opt['net_conf'])
    

if __name__ == "__main__":
    main()