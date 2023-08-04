import os

class ModelType:
    """
    A class that defines constants for different model types.
    """
    TEXT_GENERATION = "text-generation"
    ZERO_SHOT = "zero-shot-classification"
    TEXT_TEXT = "text2text-generation"
    CONVERSATIONAL = "conversational"
    TXT_CLASSIFY = "text-classification"

def create_result_file_path(file_path, extension=".txt", output_dir="./output/", suffix="model"):
    """
    Generates a new file path for a result file in the output directory.

    Args:
        file_path (str): The original file path.
        extension (str, optional): The desired file extension for the new file. Defaults to '.txt'.
        output_dir (str, optional): The directory for the new file. Defaults to './output/'.
        suffix (str, optional): The extra folder inside directory for easier segregation. Defaults to "model".

    Returns:
        str: The path for the new file or None if there was an error.
    """
    try:
        # Create the output directory if it doesn't exist
        suffix_dir = os.path.join(output_dir, suffix)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(suffix_dir):
            os.makedirs(suffix_dir)

        file_name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
        new_file_path = os.path.join(suffix_dir, file_name_without_extension + extension)
        print(f"Created new file path: {new_file_path}")
        return new_file_path
    except OSError as e:
        print(f"Error creating directory: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def generate_prompt(protocol, payload):
    """
    Generates a prompt string using the provided protocol and payload.

    Args:
        protocol (str): The protocol string.
        payload (str): The payload string.

    Returns:
        str: The generated prompt string.
    """
    return """
    Protocol: {0}
    Payload: {1}
    """.format(protocol, payload)
