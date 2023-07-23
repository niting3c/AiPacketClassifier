import argparse
import logging
import os

from excel_opearations import ExcelOperations
from models import Zero_Shot_Models
from pcap_operations import PCAP_OPERATIONS

# Suppress unnecessary scapy warnings
logging.getLogger('scapy.runtime').setLevel(logging.ERROR)


# Directory containing pcap files to be processed
def main():
    zeroShotModels = Zero_Shot_Models()
    excelOperations = ExcelOperations()
    pcap_operations = PCAP_OPERATIONS()
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
        zero_shot_models = zeroShotModels.get_models_by_name(args.model)
    elif args.suffix:
        zero_shot_models = zeroShotModels.get_models_by_suffix(args.suffix)
    else:
        zero_shot_models = zeroShotModels.get_all_models()

    for zero_shot in zero_shot_models:
        zero_shot["base_truth"] = excelOperations.read_xlsx()
        pcap_operations.process_files(zero_shot, directory)
