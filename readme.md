# Finetuned PaliGemma Inference

This repository contains scripts for running inference using a finetuned PaliGemma model. The script processes images from a specified directory, generates captions using the model, and saves each caption as a text file corresponding to the image filename.

## Repository Contents

- `paligemma_inference.py`: Main script for running inference.
- `download_tokenizer.py`: Script to download the tokenizer model.
- `requirements.txt`: List of required packages.
- `README.md`: Instructions for setting up and running the scripts.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/markuryy/finetuned-paligemma-inference.git
cd finetuned-paligemma-inference
```

### 2. Install the Required Dependencies

Make sure you have Python installed. Then, install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Download the Tokenizer and big_vision Repository

Run the `download_tokenizer.py` script to download the tokenizer model and clone the `big_vision` repository:

```bash
python download_tokenizer.py
```

## Running Inference

To run inference on a directory of images and save the generated captions:

```bash
python paligemma_inference.py --model_path /path/to/model.npz --image_dir /path/to/images --output_dir /path/to/output
```

Replace the paths with the appropriate locations for your model, image directory, and output directory.

### Example

```bash
python paligemma_inference.py --model_path ./my-custom-paligemma-ckpt-final.npz --image_dir ./images --output_dir ./captions
```

## Scripts

### paligemma_inference.py

This script loads the finetuned PaliGemma model, processes images from a specified directory, generates captions, and saves each caption as a text file.

#### Arguments

- `--model_path`: Path to the finetuned model file (e.g., `./my-custom-paligemma-ckpt-final.npz`).
- `--image_dir`: Directory containing images for inference.
- `--output_dir`: Directory to save the generated captions.

### download_tokenizer.py

This script downloads the tokenizer model required by PaliGemma. It saves the tokenizer model to `./paligemma_tokenizer.model` by default.

## Requirements

The dependencies for this project are listed in the `requirements.txt` file. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.