"""
This module provides functions for initializing, generating responses with and without classifiers from a hugging face model.
"""
from transformers import pipeline


def initialize_classifier(hugging_face_model_name, model_type):
    """
    Initializes a transformer initialised_model for the given model name.

    Args:
        hugging_face_model_name (str): The name of the transformer model.
        model_type: type of model used

    Returns:
        A transformer initialised_model for the given model name.

    """
    try:
        return pipeline(task=model_type,
                        model=hugging_face_model_name,
                        trust_remote_code=True)
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
        return initialised_model(classifier_input_str)
    except Exception as e:
        print(f"Error generating response: {e}")
        return None
