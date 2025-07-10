import os
import argparse

from dropbox_extract import RawDataDropbox
from combine_processed_data import CombineProcessedData

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dropbox", type=str,
                        default="/home/calvin/Dropbox/PersonalFolders/Calvin",
                        help="Path to the Dropbox folder")
    parser.add_argument("--raw_data_path", type=str,
                        default="/home/calvin/code/chemprop_phd_customised/habnet/data/extraction/raw",
                        help="Path to the raw data folder")
    parser.add_argument("--save_path", type=str,
                        default="/home/calvin/code/chemprop_phd_customised/habnet/data/extraction/processed/",
                        help="Path to the save folder")
    parser.add_argument("--path_to_datafiles", type=str,
                        default="/home/calvin/code/chemprop_phd_customised/habnet/data/extraction/processed/",
                        help="Path to the data files")
    return parser.parse_args()

def main(path_dropbox, raw_data_path, save_path, path_to_datafiles):
    RawDataDropbox(path_dropbox, raw_data_path, save_path)
    CombineProcessedData(path_to_datafiles)

if __name__ == "__main__":
    # If arguments are not provided, use the default values
    args = parse_args()
    main(args.path_dropbox, args.raw_data_path, args.save_path, args.path_to_datafiles)
