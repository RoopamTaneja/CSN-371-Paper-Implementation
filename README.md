# CSN-371 Artificial Intelligence

## Deepfake Video Detection Using Convolutional Vision Transformer

This repository implements a deepfake video detection system based on the Convolutional Vision Transformer architecture, proposed by [this paper](https://arxiv.org/abs/2102.11126).

Tested and run on Ubuntu (WSL).

---

## Project Structure

```bash
.
├── conv_vision_transformer
│   ├── conv_vision_transformer_model.py  # Model definition
│   ├── predict.py # Prediction script
│   ├── prediction_data/ # Portion of prediction dataset
│   ├── process_videos.py # Video preprocessing
│   ├── sample_video_dataset/ # Sample from video dataset
│   ├── train.py # Training script
│   ├── training_data/ # Portion of training dataset
│   ├── logs/ # Training and testing statistics
│   └── utils/ # Utility scripts
├── README.md
├── Report
├── Presentation
└── requirements.txt
```

---

## Setup

1. **Clone the repository**. Create a virtual environment and activate it.

2. **Install dependencies:**
    
    ```bash
    pip install -r requirements.txt
    ```

3. **Change directory:**
    
    ```bash
    cd conv_vision_transformer/
    ```

---

## Training

Training the model:

```bash
python train.py -d <path_to_training_data> -e <num_epochs>
```

`-d`: Path to the training data directory (required)

`-e`: Number of epochs (default: 1)

**Example** :

```bash
python train.py -d training_data/ -e 10
```

---

## Prediction

Running predictions:

```bash
python predict.py --model-path <path_to_model> -d <path_to_prediction_data>
```

`--model-path`: Path to the trained model file (required)

`-d`: Path to the prediction data directory (required). It should have a `metadata.json` file.

**Example** :

```bash
python predict.py --model-path saved_model/trained_model.pth -d prediction_data/
```

The prediction results are printed as well as stored in a csv file. Accuracy, log loss and AUC score are also calculated and printed.

---

For demonstrating how faces are being extracted from videos for input to the model, a sample video dataset has been included.

The functionality can be tested by running the `process_videos.py` script. 

```bash
python process_videos.py
```

---

**Key Python packages used** :

- Pytorch
- face_recognition
- Blazeface
- OpenCV
- torchvision
- albumentations
- PIL
- Scikit-learn
- Numpy

---
