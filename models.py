import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


class ZeroShotModels:
    ZERO_SHOT = "zero-shot-classification"
    ATTACK = "attack"
    NORMAL = "normal"

    candidate_labels = [
        ATTACK,
        NORMAL
    ]

    models = [
        {
            "model": None,
            "suffix": "DeBERTa",
            "model_name": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            "context_size": 800,
        },
        {
            "model": None,
            "suffix": "facebook_zero",
            "model_name": "facebook/bart-large-mnli",
            "context_size": 800,
        },
        {
            "model_name": "mosaicml/mpt-7b-8k-instruct",
            "model": None,
            "suffix": "mpt-7b",
            "context_size": 1600,
        },
        {
            "model_name": "meta-llama/Llama-2-7b-hf",
            "model": None,
            "suffix": "llama-2-7b",
            "context_size": 3500,
        },
        {
            "model_name": "meta-llama/Llama-2-13b-hf",
            "model": None,
            "suffix": "llama-2-13b",
            "context_size": 3500,
        },
        {
            "model_name": "tiiuae/falcon-7b-instruct",
            "model": None,
            "suffix": "falcon-7b",
            "context_size": 1600,
        },
        {
            "model": None,
            "suffix": "vicuna",
            "model_name": "mlc-ai/mlc-chat-vicuna-v1-7b-q4f32_0",
            "context_size": 800,
        },
    ]

    def get_models_by_suffix(self, suffix):
        return [model for model in self.models if model["suffix"] == suffix]

    def get_models_by_name(self, model_name):
        return [model for model in self.models if model["model_name"] == model_name]

    def get_all_suffixes(self):
        return list(set(model["suffix"] for model in self.models))

    def get_all_model_names(self):
        return list(set(model["model_name"] for model in self.models))

    def get_all_models(self):
        return self.models

    def initialise_model(self, hugging_face_model_name):
        try:
            print(f"Loading:{hugging_face_model_name} ")
            print(f"GPU Being Used: {torch.cuda.is_available()}")
            tokenizer = AutoTokenizer.from_pretrained(hugging_face_model_name,
                                                      cache_dir="/tmp/nitin/data_model",
                                                      trust_remote_code=True,
                                                      use_auth_token=True, )

            model = AutoModelForCausalLM.from_pretrained(hugging_face_model_name,
                                                         trust_remote_code=True,
                                                         use_auth_token=True,
                                                         device_map="auto",
                                                         cache_dir="/tmp/nitin/data_model")
            if tokenizer is None or model is None:
                print(f"Error initializing {hugging_face_model_name}")
                return None

            return pipeline(task=self.ZERO_SHOT, model=model, tokenizer=tokenizer, device_map="auto", )
        except Exception as e:
            print(f"Error initializing {hugging_face_model_name}: {e}")
            return None

    def classify(self, model, input_strings):
        if model is None:
            print("Model not initialized")
            return []
        try:
            return model(input_strings, self.candidate_labels)
        except Exception as e:
            print(f"Error generating response from classifier: {e}")
            return []
