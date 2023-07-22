import json
import os

from scapy.all import rdpcap

import llm_model
import models
import utils
from utils import create_result_file_path


class PCAP_OPERATIONS:

    def process_files(self, model_entry, directory):
        try:
            for root, dirs, files in os.walk(directory):
                for file_name in files:
                    if file_name.endswith(".pcap"):
                        file_path = os.path.join(root, file_name)
                        packet_split, result_file_path = self.analyse_packet(file_path, model_entry)
                        if packet_split is None or packet_split == 0:
                            continue
                        self.send_to_llm_model(result_file_path, model_entry, packet_split)
                        print(f"Processed: {file_path}")
        except Exception as e:
            print(f"Error processing files: {e}")

    def analyse_packet(self, file_path, model_entry):
        try:
            model_entry["input_objects"] = []
            packets = rdpcap(file_path)
            # only processing first 150 packets based on our truth base
            packets = packets[:150]

            for i, packet in enumerate(packets):
                protocol, payload = self.extract_payload_protocol(packet)
                self.prepare_input_objects(protocol, payload, model_entry, i)
            return create_result_file_path(file_path, '.txt', "./output/", model_entry["suffix"])
        except Exception as e:
            print(f"Error analysing packet: {e}")

    def extract_payload_protocol(self, packet):
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

    def send_to_llm_model(self, filepath, model_entry):
        if model_entry["model"] is None:
            print("Model Failed to initialise")
            return

        with open(filepath, "w") as output_file:
            print(f"Using ZERO_SHOT model:{model_entry['model_name']}")
            for input_object in model_entry["input_objects"]:
                prompt = utils.generate_prompt(input_object["protocol"], input_object["payload"])

                result = models.Zero_Shot_Models.classify(model_entry["model"], prompt)

            result = llm_model.pipe_response_generate_with_classifier(model_entry["model"], model_entry["input_string"])
            for entry in result:
                entry["scores"] = [score * 100 for score in entry["scores"]]
            json_result = json.dumps(result, indent=4)  # 4 spaces indentation for better visibility.
            print(json_result, file=output_file)
            output_file.flush()

    def prepare_input_objects(self, protocol, payload, model_entry, packet_num):
        try:
            if payload is None:
                print("no payload found")
            batch_size = model_entry["context_size"]
            num_batches = len(payload) // batch_size
            if len(payload) % batch_size:
                num_batches += 1
            if num_batches > 1:
                for i in range(num_batches):
                    start_index = i * batch_size
                    end_index = start_index + batch_size
                    model_entry["input_objects"].append(
                        {
                            "packet": packet_num,
                            "protocol": protocol,
                            "payload": payload[start_index:end_index],
                            "split": batch_size,
                            "current": i
                        })
            else:
                model_entry["input_objects"].append({"protocol": protocol,
                                                     "payload": payload})
        except Exception as e:
            print(f"Error sending to model: {e}")
