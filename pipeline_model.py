"""
This module provides functions for initializing, generating responses with and without classifiers from a hugging face model.
"""
import torch.cuda
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


def initialize_classifier(hugging_face_model_name, model_type, token):
    """
    Initializes a transformer initialised_model for the given model name.

    Args:
        hugging_face_model_name (str): The name of the transformer model.
        model_type: type of model used
        token: auth token
    Returns:
        A transformer initialised_model for the given model name.
    """
    try:
        print(f"Loading:{hugging_face_model_name} ")
        print(f"GPU Being Used: {torch.cuda.is_available()}")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                  cache_dir="/tmp/nitin/data_model",
                                                  trust_remote_code=True,
                                                  use_auth_token=True, )

        model = AutoModelForCausalLM.from_pretrained(hugging_face_model_name,
                                                     trust_remote_code=True,
                                                     use_auth_token=True,
                                                     device_map="auto",
                                                     cache_dir="/tmp/nitin/data_model")
        return pipeline(task=model_type, model=model, tokenizer=tokenizer, device_map="auto", )
    except Exception as e:
        print(f"Error initializing {hugging_face_model_name}: {e}")
        return None


def pipe_response_generate_without_classifier(initialised_model, classifier_input_str):
    """
    Generate a response from the model without classification for the given input string.

    Args:
        initialised_model: The model.
        classifier_input_str (str): The input string for the model.

    Returns:
        The response from the model.
    """
    try:
        return initialised_model(classifier_input_str,
                                 max_length=20,
                                 do_sample=True,
                                 top_k=10,
                                 num_return_sequences=1,
                                 )
    except Exception as e:
        print(f"Error generating response: {e}")
        return None
