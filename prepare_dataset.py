from datasets import ClassLabel

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
pcap.process_files(model, './inputs/mix_attack_data/', False, True)
data.prepare_mix_data(model["train"])
model["input_objects"] = []
pcap.process_files(model, './inputs/Non-Malicious/', False, False)
data.prepare_not_malicious_data(model["train"])

data.writeCsv("./data/mixed_data.csv")

# directly use the non-malicious one already we read
data.prepare_not_malicious_data(model["train"], False)

data.writeCsv("./data/normal.csv")
