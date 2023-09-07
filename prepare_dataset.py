import datasets
from datasets import Dataset, load_dataset, concatenate_datasets

import utils
from excel_opearations import ExcelOperations
from models import ZeroShotModels
from pcapoperations import PcapOperations


class PrepareData:
    def __init__(self):
        self.excel_opearations = ExcelOperations()
        self.base_truth = self.excel_opearations.read_xlsx()
        self.processed = []

    def prepare_mix_data(self, data):
        for obj in data:
            filename = obj["file_name"]
            result_list = self.base_truth[filename]

            for i, res in enumerate(obj["result"]):
                payload = utils.generate_prompt(res["protocol"], res["payload"])
                if result_list[i]:
                    self.processed.append({"text": payload, "label": "attack"})
                else:
                    self.processed.append({"text": payload, "label": "normal"})

    def prepare_not_malicious_data(self, data, limit=True):
        # lets keep exactly non-malicious data at 30% and 70% malicious as mix has some non-malicious
        current_payload = len(self.processed)
        count = 0
        max_limit = 0
        if limit:
            max_limit = current_payload / 3
        for obj in data:
            for res in obj["result"]:
                if res["payload"] =='unknown' or res["payload"] == '' or res["payload"] == 'unknown\n' or res[
                    "payload"] == '\n':
                    continue

                self.processed.append({"text": utils.generate_prompt(res["protocol"],
                                                                     res["payload"]),
                                       "label": "normal"})
                if limit:
                    count += 1
                    if count >= max_limit:
                        return

    def writeCsv(self, filename):
        self.excel_opearations.write_csv(self.processed, filename)


data = PrepareData()
zero_shot = ZeroShotModels()
pcap = PcapOperations()
model = zero_shot.get_models_by_suffix("llama-2-7b")[0]

# pcap.process_files(model, './inputs/Non-Malicious/', False, False)

# directly use the non-malicious one already we read
# data.prepare_not_malicious_data(model["train"], False)
# data.writeCsv("./data/normal.csv")
# candidate_labels = ["attack", "normal"]
#
candidate_labels = ["attack", "normal"]

dataset = load_dataset('csv', data_files={'test': './data/normal.csv'},features=datasets.Features(
    {'text': datasets.Value('string'), 'label': datasets.ClassLabel(num_classes=2, names=candidate_labels)}))


full_dataset = load_dataset("niting3c/Malicious_packets")

test_data = full_dataset["test"]

new_dataset = concatenate_datasets([full_dataset["test"], dataset["test"]])
full_dataset["test"] =   new_dataset

full_dataset.push_to_hub("niting3c/Malicious_packets")
