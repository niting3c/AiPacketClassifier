import argparse
import logging

from pcap_operations import process_files
from pipeline_model import initialize_classifier
from utils import ModelType

# Suppress unnecessary scapy warnings
logging.getLogger('scapy.runtime').setLevel(logging.ERROR)

# Dictionary of transformer models to be used
models = [
    # {
    #     "model": None,
    #     "str": [],
    #     "suffix": "DeBERTa",
    #     "type": ModelType.ZERO_SHOT,
    #     "model_name": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    #     "context": 800,
    #     "comment":"processed",
    #     "pipeline":True,
    # },
    # {
    #     "model": None,
    #     "str": [],
    #     "suffix": "facebook_zero",
    #     "type": ModelType.ZERO_SHOT,
    #     "model_name": "facebook/bart-large-mnli",
    #     "context": 800
    #     "pipeline":True,
    #     "comment":"processed"
    # },

    # {
    #     "model_name": "mosaicml/mpt-7b-8k-instruct",
    #     "model": None,
    #     "pipeline": False,
    #     "str": [],
    #     "suffix": "mpt-7b",
    #     "type": ModelType.TEXT_GENERATION,
    #     "context": 1600,
    #     "comment": "did not work on cpu and pipeline"
    # },
    {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "model": None,
        "pipeline": False,
        "str": [],
        "suffix": "llama-2-7b",
        "type": ModelType.TEXT_GENERATION,
        "context": 3500,
    },
    {
        "model_name": "tiiuae/falcon-7b-instruct",
        "model": None,
        "pipeline":False,
        "str": [],
        "suffix": "falcon-7b",
        "type": ModelType.TEXT_GENERATION,
        "context": 1600,
    },
    {
        "model_name": "meta-llama/Llama-2-13b-hf",
        "model": None,
        "pipeline": False,
        "str": [],
        "suffix": "llama-2-7b-zero",
        "type": ModelType.ZERO_SHOT,
        "context": 3500,
    },
    {
        "model_name": "tiiuae/falcon-7b-instruct",
        "model": None,
        "pipeline": False,
        "str": [],
        "suffix": "falcon-7b-zero",
        "type": ModelType.ZERO_SHOT,
        "context": 1600,
    },
    # {
    #     "model": None,
    #     "str": [],
    #     "pipeline":False,
    #     "suffix": "vicuna",
    #     "type": ModelType.TEXT_GENERATION,
    #     "model_name": "mlc-ai/mlc-chat-vicuna-v1-7b-q4f32_0",
    #     "context": 800,
    #     "comment": "did not work on cpu"
    # },
]

# Directory containing pcap files to be processed
directory = './inputs'

argParser = argparse.ArgumentParser()
argParser.add_argument("-t", "--token", help="Huggingface Auth Token")

args = argParser.parse_args()

# Process the pcap files for each model
for i, entry in enumerate(models):
    entry["model"] = initialize_classifier(entry["model_name"], entry["type"], args.token)
    process_files(directory, entry)
    models[i] = {}
