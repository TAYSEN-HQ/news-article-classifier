# News Article Classifier

![GitHub repo size](https://img.shields.io/github/repo-size/suwilanji-chipofya-hadat/news-article-classifier)
![GitHub stars](https://img.shields.io/github/stars/suwilanji-chipofya-hadat/news-article-classifier?style=social)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![License](https://img.shields.io/badge/license-GNU%20GPL%20v3-blue)

This repository contains a news classification model built using a Transformer-based architecture implemented using TensorFlow and Keras. The model is trained to classify news articles into different categories using the News_Category_Dataset_v3 dataset from Kaggle.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates the application of a transformer-based model in the field of news classification. Transformers have shown exceptional performance in natural language processing tasks, making them well-suited for this classification task.

## Dataset

The model is trained and evaluated using the [News_Category_Dataset_v3](https://www.kaggle.com/rmisra/news-category-dataset) dataset from Kaggle. The dataset contains news articles along with their corresponding categories, which we will use for training and testing our model.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/suwilanji-chipofya-hadat/news-article-classifier.git
   ```

2. Navigate to the project directory:
   ```bash
   cd news-article-classifier
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the [News_Category_Dataset_v3](https://www.kaggle.com/rmisra/news-category-dataset) dataset and place it in the `data/` directory.

2. To train the model, run the following command:
   ```bash
   python model.py
   ```

3. Use the trained model to classify news articles:

use the generate function in `generate.py` to generate predictions

## Model Architecture

The model architecture is based on the Transformer model, which has proven to be highly effective for various NLP tasks. It consists of an encoder stack with self-attention mechanisms, enabling it to capture contextual relationships within the input text effectively.

## Training

The model is trained using the TensorFlow and Keras libraries. During training, the text data is tokenized, and the target categories are one-hot encoded. The model is trained using a suitable loss function and optimizer.

## Contributing

Contributions to this project are welcome! If you have any improvements or suggestions, please open an issue or a pull request.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).

---
