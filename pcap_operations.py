import os

from scapy.all import rdpcap
from transformers import Conversation

import PromptMaker
import llm_model
from llm_model import prepare_input_strings
from utils import ModelType
from utils import create_result_file_path


def process_files(directory, model_entry):
    """
    Processes all pcap files in the specified directory.

    Args:
        directory (str): The directory containing pcap files.
    """
    try:
        for root, dirs, files in os.walk(directory):
            for file_name in files:
                if file_name.endswith(".pcap"):
                    file_path = os.path.join(root, file_name)
                    model_entry["str"] = []
                    model_entry["chat"] = None
                    result_file_path = analyse_packet(file_path, model_entry)
                    send_to_llm_model(result_file_path, model_entry)
                    print(f"Processed: {file_path}")
    except Exception as e:
        print(f"Error processing files: {e}")


def analyse_packet(file_path, model_entry):
    """
    Analyzes a pcap file and extracts packet information for further processing.

    Args:
        file_path (str): The path to the pcap file.
        classifier: The classifier model.
    """
    try:
        # Create a path for the result file
        model_entry["str"] = []
        packets = rdpcap(file_path)
        for packet in packets:
            protocol, payload = extract_payload_protocol(packet)
            prepare_input_strings(protocol, payload, model_entry)
        return create_result_file_path(file_path, '.txt', "./output/", model_entry["suffix"])
    except Exception as e:
        print(f"Error analysing packet: {e}")


def extract_payload_protocol(packet):
    """
    Extracts payload and protocol from the packet.

    Args:
        packet: The packet to process.

    Returns:
        tuple: The payload and protocol.
    """
    try:
        payload = repr(packet.payload)
        if packet.haslayer('IP'):
            protocol = "IP"
        elif packet.haslayer('TCP'):
            if packet.payload.haslayer('FTP'):
                protocol = "TCP"
            else:
                protocol = "TCP"
        elif packet.haslayer('UDP'):
            protocol = "UDP"
        elif packet.haslayer('ICMP'):
            protocol = "ICMP"
        else:
            protocol = "unknown"
        return protocol, payload
    except AttributeError:
        print("Error: Attribute not found in the packet.")
    except Exception as e:
        print(f"Error extracting payload and protocol: {e}")

    return "", ""


def send_to_llm_model(filepath, model_entry):
    model_type = model_entry["type"]
    if model_entry["model"] is None:
        print("Model Failed to initialise")
        return
    with open(filepath, "w") as output_file:
        match model_type:
            case ModelType.CONVERSATIONAL:
                process_conversational_model(model_entry, output_file)
            case ModelType.ZERO_SHOT:
                process_zero_shot_model(model_entry, output_file)
            case _:
                process_other_model(model_entry, output_file, model_type)
        output_file.flush()


def process_conversational_model(model_entry, output_file):
    print(f"Using Conversational model:{model_entry['model_name']}", file=output_file)
    model_entry["chat"] = Conversation(PromptMaker.generate_first_prompt(len(model_entry["str"])))

    # send conversations to model
    result = model_entry["model"](model_entry["chat"])
    print(f"Conversations processed:{str(result)}", file=output_file)

    for entry in model_entry["str"]:
        model_entry["chat"].add_user_input(entry)
        result = model_entry["model"](model_entry["chat"])
        print(f"Conversations processed:{str(result)}", file=output_file)

    model_entry["chat"].add_user_input(PromptMaker.generate_text_chat_last_prompt())
    result = model_entry["model"](model_entry["chat"])
    print(f"Conversations processed:{str(result)}", file=output_file)


def process_zero_shot_model(model_entry, output_file):
    print(f"Using ZERO_SHOT model:{model_entry['model_name']}", file=output_file)
    result = llm_model.pipe_response_generate_with_classifier(model_entry["model"], model_entry["str"])
    print(f"Result from the packet file:{str(result)}\n", file=output_file)


def process_other_model(model_entry, output_file, model_type):
    print(f"Using {model_entry['type']} model:{model_entry['model_name']}", file=output_file)
    for entry in model_entry["str"]:
        print("----" * 40, file=output_file)
        print(entry + "\n\n", file=output_file)
        result = llm_model.pipe_response_generate_without_classifier(model_entry["model"], entry)
        if model_type == ModelType.ZERO_SHOT:
            print(f"Result from the packet file:{str(result['scores'])}", file=output_file)
        else:
            print(f"Result from the packet file:{str(result)}", file=output_file)
        print("----" * 40, file=output_file)

# Example usage:
# classifier = create_pipeline_model()
# process_files("./inputs/", None, "hi")
