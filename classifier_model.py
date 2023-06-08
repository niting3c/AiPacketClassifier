"""
This module provides functions for generating responses with classifier from a hugging face model.
"""
candidate_labels = ['malicious', 'not malicious', 'attack']

def pipe_response_generate_with_classifier(initialised_model, classifier_input_str):
    """
    Generate a response from the classifier for the given input string.

    Args:
        initialised_model: The classifier model.
        classifier_input_str (str): The input string for the classifier.

    Returns:
        The response from the classifier.
    """
    try:
        return initialised_model(classifier_input_str, candidate_labels)
    except Exception as e:
        print(f"Error generating response from classifier: {e}")
        return None
