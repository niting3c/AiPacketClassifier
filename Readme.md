# Training AI for Information Security: MSC Dissertation Project

This repository contains the implementation of my MSC Dissertation project on "Training AI for Information Security." The project utilizes machine learning algorithms for detecting and classifying cyber threats in network traffic, specifically employing transformer-based models for zero-shot classification tasks.

## Table of Contents

- [Installation](#installation)
- [Detailed File Descriptions](#detailed-file-descriptions)
- [Usage](#usage)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

To install the project, follow these steps:

1. Clone the repository:
    ```
    git clone https://github.com/niting3c/AiPacketClassifier.git
    ```
2. Change directory to the cloned repository:
    ```
    cd AiPacketClassifier
    ```
3. Install Conda if you haven't done so already. You can download it from [here](https://docs.conda.io/en/latest/miniconda.html).
4. Create a Conda environment using the provided `environment.yml` file:
    ```
    conda env create -f environment.yml
    ```
5. Activate the Conda environment:
    ```
    conda activate AiPacketClassifier
    ```

**Note**: This project has been tested on Python 3.9.5, and the required dependencies are listed in the `environment.yml` file.

## Detailed File Descriptions

Here are detailed descriptions of the main files in this repository:

1. `run.py`: This is the main script that initializes multiple zero-shot classification models from the Transformers library, processes input files with each model, and writes the results. It uses the following functions:
    - `load_models()`: Loads the transformer models specified in the `models.py` file and initializes the zero-shot classifiers.
    - `process_files(model_entry, directory)`: Processes pcap files in the given `directory` using the specified `model_entry`. This function calls `analyse_packet()` and `send_to_llm_model()` for each pcap file.

2. `utils.py`: This script contains helper functions to handle file-related operations such as creating file paths. It provides the following functions:
    - `create_result_file_path(file_path, extension=".txt", output_dir="./output/", suffix="model")`: Generates a new file path for a result file in the output directory. The `file_path` parameter specifies the original file path, `extension` specifies the desired file extension for the new file, `output_dir` specifies the directory for the new file (default is "./output/"), and `suffix` specifies the extra folder inside the directory for easier segregation (default is "model").
    - `get_file_path(root, file_name)`: Generates a file path by combining the provided `root` and `file_name`.

3. `promptmaker.py`: This script includes functions that generate prompts for the classification tasks. These prompts help guide the AI in its analysis of packets and instruct it on how to report its findings. It provides the following function:
    - `generate_prompt(protocol, payload)`: Generates a formatted prompt with the specified `protocol` and `payload` to be used as input for the transformer models.

4. `pcapoperations.py`: This script contains functions that handle pcap file operations, including reading packets from pcap files, analyzing packets using the zero-shot classification models, and writing the results to an output file. It provides the following functions:
    - `process_files(model_entry, directory)`: Processes pcap files in the given `directory` using the specified `model_entry`. This function calls `analyse_packet()` and `send_to_llm_model()` for each pcap file.
    - `analyse_packet(file_path, model_entry)`: Analyzes the packets in the pcap file located at `file_path` using the specified `model_entry`. This function extracts the protocol and payload from each packet and prepares input objects for classification.
    - `extract_payload_protocol(packet)`: Extracts the payload and protocol from the `packet`.
    - `send_to_llm_model(model_entry, file_name)`: Sends the prepared input objects to the ZeroShot model for classification and stores the results in the `model_entry`.

5. `llm_model.py`: This script includes functions that handle the interaction with the transformer models. It prepares the inputs for the classifier, generates the classifier's response, and processes the response.

## Usage

1. Make sure you have installed all necessary packages and activated the Conda environment (see [Installation](#installation)).

2. The `run.py` script expects input files to be located in the `./inputs` directory. Make sure you have populated this directory with your pcap files for processing.

3. To start the program, simply run:
    ```
    python run.py
    ```
4. The results will be written to the `./output` directory.

## Models

The project uses the following transformer models for zero-shot classification tasks:

1. [Deep Night Research's ZSC Text](https://huggingface.co/deepnight-research/zsc-text)
2. [Facebook's BART Large MNLI](https://huggingface.co/facebook/bart-large-mnli)
3. [Moritz Laurer's DeBERTa v3 base MNLI+FEVER+ANLI](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
4. [Sileod's DeBERTa v3 base tasksource NLI](https://huggingface.co/sileod/deberta-v3-base-tasksource-nli)

## Contributing

Contributions are what make the open-source

 community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Nitin Gupta - nitin.gupta.22@ucl.ac.uk

Project Link: [https://github.com/niting3c/AiPacketClassifier](https://github.com/niting3c/AiPacketClassifier)

For specific requests or inquiries, feel free to contact me. Happy coding!

---

In this updated README file, I have provided more detailed explanations for each section, including function details and their usages. If you need any further improvements or additional information, please let me know!