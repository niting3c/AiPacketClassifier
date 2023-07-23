# Training AI for Information Security: MSC Dissertation Project

This repository contains the implementation of my MSC Dissertation project on "Training AI for Information Security". The project uses machine learning algorithms for detecting and classifying cyber threats in network traffic, specifically utilizing transformer-based models for zero-shot classification tasks.

## Table of Contents

- [Installation](#installation)
- [Detailed File Descriptions](#detailed-file-descriptions)
- [Usage](#usage)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

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

**Note**: This project has been tested on Python 3.9.5 and the required dependencies are listed in the `environment.yml` file.

## Detailed File Descriptions

Here are brief descriptions of the main files in this repository:

1. `run.py`: This is the main script that initializes multiple zero-shot classification models from the Transformers library, processes input files with each model, and writes the results.

2. `utils.py`: This script contains helper functions to handle file-related operations such as creating file paths.
    - `create_result_file_path`: This function creates a result file path for the output files based on the original file path and the specified extension and directory.
    - `get_file_path`: This function generates a file path by combining the provided root and file name.

3. `promptmaker.py`: This script includes functions that generate prompts for the classification tasks. These prompts help guide the AI in its analysis of packets and instruct it on how to report its findings.

4. `pcapoperations.py`: This script contains functions that handle pcap file operations, including reading packets from pcap files, analyzing packets using the zero-shot classification models, and writing the results to an output file.

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

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Nitin Gupta - nitin.gupta.22@ucl.ac.uk

Project Link: [https://github.com/niting3c/AiPacketClassifier](https://github.com/niting3c/AiPacketClassifier)
  
For specific requests or inquiries, feel free to contact me. Happy coding!
