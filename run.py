import argparse
import logging
import os
from pcap_operations import PCAP_OPERATIONS
from models import Zero_Shot_Models

# Suppress unnecessary scapy warnings
logging.getLogger('scapy.runtime').setLevel(logging.ERROR)


# Directory containing pcap files to be processed
def main():
    directory = './inputs'
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-t", "--token", help="Huggingface Auth Token")
    argParser.add_argument("-m", "--model", help="Model Name")
    argParser.add_argument("-s", "--suffix", help="Model Suffix")
    argParser.add_argument("-d", "--directory", help="Directory containing pcap files to be processed")
    args = argParser.parse_args()


    if args.directory:
        directory = args.directory

    if args.token:
        os.environ['HF_TOKEN'] = args.token

    if args.model:
        zero_shot_models = Zero_Shot_Models.get_models_by_name(args.model)
    elif args.suffix:
        zero_shot_models = Zero_Shot_Models.get_models_by_suffix(args.suffix)
    else:
        zero_shot_models = Zero_Shot_Models.get_all_models()


    for zero_shots in zero_shot_models:
        PCAP_OPERATIONS.process_files(zero_shots, directory)