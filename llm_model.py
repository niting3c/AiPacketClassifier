"""
This module provides functions for processing the input string with the classifier and write the result to output file,
sending the protocol and payload to the model for classification.
"""
from PromptMaker import generate_prompt, generate_part_prompt, generate_part_prompt_final
from classifier_model import pipe_response_generate_with_classifier
from pipeline_model import pipe_response_generate_without_classifier
from utils import ModelType


def process_string_input(input_string, model_entry, outputfile):
    """
    Processes the input string with the classifier and writes the result to an output file.

    Args:
        input_string (str): The input string for the classifier.
        model_entry: Model entry with all the references
        outputfile: The output file.
    """
    print("-----" * 40, file=outputfile)
    try:
        match model_entry["type"]:
            case ModelType.CONVERSATIONAL:
                print(f"\nInput:{input_string}\n", file=outputfile)
                conversation_input = model_entry["chat"].add_user_input(input_string)
                model_entry["chat"] = conversation_input
                result = model_entry["model"](conversation_input)
            case ModelType.ZERO_SHOT:
                result = pipe_response_generate_with_classifier(model_entry["model"], input_string)
            case _:
                result = pipe_response_generate_without_classifier(model_entry["model"], input_string)
        print(f"\nString processed with result = {str(result)}", file=outputfile)
        print("-----" * 40, file=outputfile)
    except Exception as e:
        print(f"Error processing string input: {e}")


def prepare_input_strings(protocol, payload, model_entry, index):
    """
    Sends the protocol and payload to the model for classification.

    Args:
        protocol (str): The protocol.
        payload: The payload.
        model_entry: The model_entry model.
    """
    try:
        if payload is None:
            print("no payload found")
            model_entry["str"].append(generate_prompt(protocol, "", index))
            return
        batch_size = model_entry["context"]
        num_batches = len(payload) // batch_size
        if len(payload) % batch_size:
            num_batches += 1
        if num_batches > 1:
            for i in range(num_batches):
                start_index = i * batch_size
                end_index = start_index + batch_size
                model_entry["str"].append(
                    generate_part_prompt(protocol,
                                         payload[start_index:end_index],
                                         i + 1,
                                         num_batches, index))
            model_entry["str"].append(generate_part_prompt_final())
        else:
            model_entry["str"].append(generate_prompt(protocol, payload, index))
    except Exception as e:
        print(f"Error sending to model: {e}")
