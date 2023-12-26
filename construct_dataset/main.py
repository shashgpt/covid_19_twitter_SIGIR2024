import argparse
import sys
import pprint
from datetime import datetime

from scripts.construct_covid19_dataset import Construct_covid19_dataset

if __name__=='__main__':

    #gather parser arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_name",
                        type=str,
                        required=True)
    parser.add_argument("--start",
                        type=int,
                        required=True)
    parser.add_argument("--stop",
                        type=int,
                        required=True)
    args = parser.parse_args()
    config = vars(args)
    print("\n")
    pprint.pprint(config)
    print("\n")

    Construct_covid19_dataset(config).create_dataset()
