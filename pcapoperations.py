import os

from scapy.all import rdpcap

import utils
from models import ZeroShotModels


class PcapOperations:
    POSITIVE = "Positive"
    FALSE_POSITIVE = "False Positive"
    FALSE_NEGATIVE = "False Negative"
    NEGATIVE = "Negative"
    zeroShotModels = ZeroShotModels()

    def process_files(self, model_entry, directory):
        """
        Process all .pcap files in the given directory using the provided model_entry.

        Args:
            model_entry (dict): The model entry containing model details.
            directory (str): The directory containing the .pcap files to process.
        """
        try:
            model_entry["model_output"] = {
                "name": model_entry["suffix"],
                "items": []
            }

            for root, dirs, files in os.walk(directory):
                for file_name in files:
                    if file_name.endswith(".pcap"):
                        file_path = os.path.join(root, file_name)
                        self.analyse_packet(file_path, model_entry)
                        self.send_to_llm_model(model_entry, os.path.splitext(os.path.basename(file_path))[0])
                        print(f"Processed: {file_path}")
        except Exception as e:
            print(f"Error processing files: {e}")

    def analyse_packet(self, file_path, model_entry):
        """
        Analyze the packets in the .pcap file and prepare input objects.

        Args:
            file_path (str): The path to the .pcap file.
            model_entry (dict): The model entry containing model details.
        """
        try:
            model_entry["input_objects"] = []
            packets = rdpcap(file_path)
            # only processing first 150 packets based on our truth base
            packets = packets[:150]

            for i, packet in enumerate(packets):
                protocol, payload = self.extract_payload_protocol(packet)
                self.prepare_input_objects(protocol, payload, model_entry, i)
        except Exception as e:
            print(f"Error analyzing packet: {e}")

    def extract_payload_protocol(self, packet):
        """
        Extract the payload and protocol from the packet.

        Args:
            packet: The packet to extract payload and protocol from.

        Returns:
            str: The protocol name.
            str: The payload representation.
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
            return "", ""
        except Exception as e:
            print(f"Error extracting payload and protocol: {e}")
            return "", ""

    def send_to_llm_model(self, model_entry, file_name):
        """
        Send the prepared input objects to the ZeroShot model.

        Args:
            model_entry (dict): The model entry containing model details.
            file_name (str): The name of the file being processed.
        """
        if model_entry["model"] is None:
            print("Model Failed to initialize")
            return
        item = {"file_name": file_name, "result": []}
        base_truth = model_entry["base_truth"][file_name]  # get the base truth for the file
        print(f"Using ZERO_SHOT model: {model_entry['model_name']}")
        batched_result_aggregation = []
        for input_object in model_entry["input_objects"]:

            packet_num = input_object.get("packet_num", None)
            protocol = input_object.get("protocol", None)
            payload = input_object.get("payload", None)
            split = input_object.get("split", None)
            batched = input_object.get("batched", False)

            prompt = utils.generate_prompt(protocol, payload)
            classify_result = self.zeroShotModels.classify(model_entry["model"], prompt)

            # check the scores corresponding to the labels
            scores = classify_result.get("scores", [])
            labels = classify_result.get("labels", [])

            normal_score = 0
            attack_score = 0

            for i, label in enumerate(labels):
                if label == self.zeroShotModels.NORMAL:
                    normal_score = scores[i]
                elif label == self.zeroShotModels.ATTACK:
                    attack_score = scores[i]

            if batched:
                batched_result_aggregation.append((normal_score, attack_score))

                if len(batched_result_aggregation) == split:
                    normal_score = sum([x[0] for x in batched_result_aggregation]) / split
                    attack_score = sum([x[1] for x in batched_result_aggregation]) / split
                    batched_result_aggregation = []
                else:
                    continue

            attack = False
            actual = False

            if attack_score > normal_score:
                attack = True
                actual = base_truth[packet_num]

            if attack and actual:
                item["result"].append(self.POSITIVE)
            elif attack and not actual:
                item["result"].append(self.FALSE_POSITIVE)
            elif not attack and actual:
                item["result"].append(self.FALSE_NEGATIVE)
            else:
                item["result"].append(self.NEGATIVE)

        model_entry["model_output"]["items"].append(item)

    def prepare_input_objects(self, protocol, payload, model_entry, packet_num):
        """
        Prepare input objects for each packet based on the model's context size.

        Args:
            protocol (str): The protocol of the packet.
            payload (str): The payload of the packet.
            model_entry (dict): The model entry containing model details.
            packet_num (int): The packet number.
        """
        try:
            if payload is None:
                print("No payload found")
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
                            "packet_num": packet_num,
                            "protocol": protocol,
                            "payload": payload[start_index:end_index],
                            "split": batch_size,
                            "batched": True,
                        })
            else:
                model_entry["input_objects"].append({"protocol": protocol,
                                                     "packet_num": packet_num,
                                                     "payload": payload})
        except Exception as e:
            print(f"Error sending to model: {e}")
