import torch
from transformers import pipeline


class ZeroShotModels:
    """
       A class to manage ZeroShot models.

       Attributes:
           ZERO_SHOT (str): Constant for ZeroShot classification.
           ATTACK (str): Constant for the "attack" candidate label.
           NORMAL (str): Constant for the "normal" candidate label.
           candidate_labels (list): List of candidate labels.
           models (list): List of model configurations.

       Methods:
           get_models_by_suffix(suffix): Returns models with the given suffix.
           get_models_by_name(model_name): Returns models with the given model name.
           get_all_suffixes(): Returns a list of all unique suffixes in the models.
           get_all_model_names(): Returns a list of all unique model names in the models.
           get_all_models(): Returns all models configurations.
           initialise_model(hugging_face_model_name): Initializes and returns a model using the given Hugging Face model name.
           classify(model, input_strings): Classifies the input strings using the provided model.

       """

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
            "model_name": "mosaicml/mpt-7b",
            "model": None,
            "suffix": "mpt-7b",
            "context_size": 1600,
        },
        {
            "model_name": "mosaicml/mpt-7b-instruct",
            "model": None,
            "suffix": "mpt-7b-2",
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
        }, {
            "model_name": "lmsys/vicuna-7b-v1.3",
            "model": None,
            "suffix": "falcon-7b",
            "context_size": 1600,
        },
        {
            "model": None,
            "suffix": "vicuna",
            "model_name": "lmsys/vicuna-7b-v1.3",
            "context_size": 800,
        },

    ]

    def get_models_by_suffix(self, suffix):
        """
        Returns models with the given suffix.

        Args:
            suffix (str): The suffix to filter models.

        Returns:
            list: List of models with the matching suffix.
        """
        return [model for model in self.models if model["suffix"].lower() == suffix.lower()]

    def get_models_by_name(self, model_name):
        """
        Returns models with the given model name.

        Args:
            model_name (str): The model name to filter models.

        Returns:
            list: List of models with the matching model name.
        """
        return [model for model in self.models if model["model_name"].lower() == model_name.lower()]

    def get_all_suffixes(self):
        """
        Returns a list of all unique suffixes in the models.

        Returns:
            list: List of all unique suffixes in the models.
        """
        return list(set(model["suffix"] for model in self.models))

    def get_all_model_names(self):
        """
        Returns a list of all unique model names in the models.

        Returns:
            list: List of all unique model names in the models.
        """
        return list(set(model["model_name"] for model in self.models))

    def get_all_models(self):
        """
        Returns all models configurations.

        Returns:
            list: List of all models configurations.
        """
        return self.models

    def initialise_model(self, hugging_face_model_name):
        """
        Initializes and returns a model using the given Hugging Face model name.

        Args:
            hugging_face_model_name (str): The Hugging Face model name.

        Returns:
            pipeline: The initialized pipeline object for the model or None if there was an error.
        """
        try:
            print(f"Loading: {hugging_face_model_name}")
            print(f"GPU Being Used: {torch.cuda.is_available()}")

            return pipeline(task=self.ZERO_SHOT,
                            model=hugging_face_model_name,
                            use_auth_token=True,
                            trust_remote_code=True,
                            )
        except Exception as e:
            print(f"Error initializing {hugging_face_model_name}: {e}")
            return None

    def classify(self, model, input_strings):
        """
        Classifies the input strings using the provided model.

        Args:
            model (pipeline): The initialized pipeline model.
            input_strings (str or list): The input string(s) to classify.

        Returns:
            score object: The classification results
        """
        if model is None:
            print("Model not initialized")
            return {}
        try:
            return model(input_strings, self.candidate_labels)
        except Exception as e:
            print(f"Error generating response from classifier: {e}")
            return {}
