# S2 Advanced Statistical Methods Coursework - Antikythera mechanism

This repository contains code, data and report for the S2 coursework on the topic of Antikythera mechanism. 

## Setting up the environment 

### Prerequisites

- Python 3.9.6
- `pip` (Python package installer)

### Setting Up a Virtual Environment

1. Open a terminal and navigate to root of this directory:


2. Create a virtual environment:
    try:
    ```sh
    python -m venv venv
    ```
    if above does not work: 
    ```sh
    python3 -m venv venv
    ```

3. Activate the virtual environment:

    - On macOS and Linux:

        ```sh
        source venv/bin/activate
        ```

    - On Windows:

        ```sh
        .\venv\Scripts\activate
        ```

### Installing Dependencies

Once the virtual environment is activated, install the required packages using `pip`:

```sh
pip install -r requirements.txt
```


## Usage

All commands below should be executed at the root of this directory. 


The code that produces figure in (a) can be executed by:

```sh
python code/visualize.py
```

The model that the referenced paper described is inside `code/antikythera_model.py`. To use the model to generate predicted locations, please execute the following command:
```sh
python code/antikythera_model.py
```

The program that computes maximum likelihood paramter is `code/optimize_antikythera.py`. To regenerate the results, please run the following command:
```sh
python code/optimize_antikythera.py
```

The HMC sampling is computed using `code/hmc.py`. To execute:
```sh
python code/hmc.py
```

The final comparison of the models uses the script `code/comparison.py`.
```sh
python code/comparison.py
```

### Declaration
This repository was developed with the assistance from Anthropic’s Claude 3.5 Sonnet. Specifically, Claude was used to assist with debugging Python code implementations. All AI-generated suggestions were manually reviewed.

