import logging

from pcap_operations import process_files
from pipeline_model import initialize_classifier
from utils import ModelType

# Suppress unnecessary scapy warnings
logging.getLogger('scapy.runtime').setLevel(logging.ERROR)

# Dictionary of transformer models to be used
models = [

    {
        "model": None,
        "str": [],
        "suffix": "vicuna",
        "type": ModelType.TEXT_GENERATION,
        "model_name": "Ejafa/vicuna_7B_vanilla_1.1",
        "context": 800,
        "comment": "did not work on cpu"
    },
    {
        "model": None,
        "str": [],
        "suffix": "koala",
        "type": ModelType.TEXT_GENERATION,
        "model_name": "TheBloke/koala-7B-HF",
        "context": 800,
        "comment" : "did not work on cpu"
    },
    {
        "model": None,
        "str": [],
        "suffix": "DeBERTa",
        "type": ModelType.ZERO_SHOT,
        "model_name": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "context": 800
    },
    {
        "model": None,
        "str": [],
        "suffix": "SecBERT",
        "type": ModelType.TEXT_GENERATION,
        "model_name": "jackaduma/SecBERT",
        "context": 800
    },
    {
        "model": None,
        "str": [],
        "suffix": "intrusion",
        "type": ModelType.TXT_CLASSIFY,
        "model_name": "rdpahalavan/bert-network-packet-flow-header-payload",
        "context": 400
    }
]

# Directory containing pcap files to be processed
directory = './inputs'

# Process the pcap files for each model
for i, entry in enumerate(models):
    entry["model"] = initialize_classifier(entry["model_name"], entry["type"])
    process_files(directory, entry)
    models[i] = {}
