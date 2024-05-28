# Road-SIgns-Classification


## Introduction

Traffic sign classification is a crucial task in autonomous driving and advanced driver-assistance systems (ADAS). Accurate identification of traffic signs enables vehicles to make informed decisions, enhancing safety and compliance with traffic regulations. This project aims to develop and evaluate various convolutional neural network (CNN) architectures to classify traffic signs based on their shapes and types.

This repository provides the complete code for data preprocessing, model training, evaluation, and inference. It includes implementations of multiple CNN architectures, data augmentation techniques, and hyperparameter tuning methods to achieve optimal performance. By leveraging these models, this project seeks to contribute to the development of more reliable and efficient traffic sign recognition systems.
```

Integrating this into the main `README.md`:

```markdown
# Traffic Signs Classification

## Introduction

Traffic sign classification is a crucial task in autonomous driving and advanced driver-assistance systems (ADAS). Accurate identification of traffic signs enables vehicles to make informed decisions, enhancing safety and compliance with traffic regulations. This project aims to develop and evaluate various convolutional neural network (CNN) architectures to classify traffic signs based on their shapes and types.

This repository provides the complete code for data preprocessing, model training, evaluation, and inference. It includes implementations of multiple CNN architectures, data augmentation techniques, and hyperparameter tuning methods to achieve optimal performance. By leveraging these models, this project seeks to contribute to the development of more reliable and efficient traffic sign recognition systems.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Augmentation](#data-augmentation)
- [Models](#models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Inference](#inference)
- [Results](#results)

## Directory Structure
The directory structure of the project is as follows:

```
trafficsigns_dataset/
├── diamond/
│   └── rightofway/
│       └── *.png
├── hex/
│   └── stop/
│       └── *.png
├── round/
│   ├── bicycle/
│   ├── limitedtraffic/
│   ├── noentry/
│   ├── noparking/
│   ├── roundabout/
│   ├── speed/
│   ├── trafficdirective/
│   └── traveldirection/
│       └── *.png
├── square/
│   ├── continue/
│   ├── crossing/
│   └── parking/
│       └── *.png
└── triangle/
    ├── giveway/
    └── warning/
        └── *.png
```

## Installation
To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/trafficsigns_classification.git
cd trafficsigns_classification
pip install -r requirements.txt
```

## Dataset
The dataset consists of traffic sign images organized by their shapes and types. Each image is in `.png` format.

## Data Augmentation
Data augmentation is performed to increase the size of the training dataset and improve the generalization of the models. The following augmentations are applied:
- Horizontal flips
- Zoom in/out
- Random cropping

## Models
Five different CNN architectures are implemented and compared:
1. **Basic Model**: A simple CNN with one convolutional layer.
2. **Basic Model with Dropout**: Similar to the basic model but with dropout layers to reduce overfitting.
3. **Deeper Model with Batch Normalization**: A deeper CNN with batch normalization layers.
4. **LeNet-5 Inspired Model**: A model inspired by the LeNet-5 architecture.
5. **Inception-like Model**: A small Inception-like model for learning spatial hierarchies.

## Training
The models are trained on the dataset for both shape and type classification problems. The training process includes:
- Splitting the data into training and testing sets.
- Training each model and recording their performance.
- Plotting validation accuracy, validation loss, and training time for comparison.

## Evaluation
The evaluation includes:
- Plotting the training and validation loss/accuracy.
- Displaying the confusion matrix.
- Visualizing model predictions.

## Hyperparameter Tuning
Hyperparameter tuning is performed on the best-performing model using Keras Tuner. The hyperparameters tuned include:
- Learning rate
- Optimizer (Adam, SGD)

## Inference
The project includes functions for preprocessing images and performing inference using the trained models.

## Results
The results of the models are compared based on validation accuracy, validation loss, and training time. The best-performing model is further fine-tuned using hyperparameter tuning.

## Usage

### Training
To train the models, run the following script:
```bash
python train.py
```

### Evaluation
To evaluate the trained models and visualize the results, run:
```bash
python evaluate.py
```

### Inference
To perform inference on a new image, use the following script:
```bash
python inference.py --image_path path/to/your/image.png
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```

This comprehensive `README.md` should provide clear guidance to users on how to utilize your traffic signs classification project.
