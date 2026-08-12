# Fashion Item Classifier (CNN + Streamlit)

A Convolutional Neural Network (CNN) that classifies clothing items into 10 categories, trained on the Fashion MNIST dataset. Deployed as an interactive Streamlit web app where users can upload an image of a clothing item and get a live prediction.

## What It Does

Upload an image of a clothing item (t-shirt, trouser, dress, shoe, bag, etc.), and the model predicts which of the 10 categories it belongs to, along with a confidence score.

## Categories

T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## Skills Demonstrated

- Convolutional Neural Networks (Conv2D, MaxPooling2D)
- Regularization with Dropout to reduce overfitting
- Image preprocessing pipeline (grayscale conversion, color inversion, cropping, normalization, reshaping)
- Multi-class classification (Softmax activation, Sparse Categorical Crossentropy loss)
- Model experimentation and comparison (tested different architectures and epoch counts to find the best-performing configuration)
- Model deployment with Streamlit
- Handling real-world input variation (user-uploaded images differ significantly from clean training data)

## Model Performance

| Experiment | Test Accuracy | Overfitting Gap |
|---|---|---|
| 2 Conv layers, 10 epochs | 91.27% | 3.61% |
| 2 Conv layers, 7 epochs | 90.80% | 3.05% |
| 3 Conv layers (32-64-128) | 89.23% | 4.00% |
| **2 Conv layers + Dropout(0.3)** | **91.65%** | **1.41%** |

The final model (2 convolutional layers with Dropout regularization) was selected as it achieved the best test accuracy with the smallest train/test gap, indicating good generalization.

## Architecture

```
Input (28x28x1)
  -> Conv2D(32, 3x3, relu) -> MaxPooling2D(2x2)
  -> Conv2D(64, 3x3, relu) -> MaxPooling2D(2x2)
  -> Flatten
  -> Dense(64, relu) -> Dropout(0.3)
  -> Dense(10, softmax)
```

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — Streamlit application (upload, preprocess, predict, display results)
- `fashion_model.keras` — Trained CNN model
- `requirements.txt` — Python dependencies

## Notes on Preprocessing

User-uploaded images rarely match the clean, centered, 28x28 grayscale format of the training data. The app handles this by:
1. Converting to grayscale
2. Inverting colors if the background is light (training data uses white-on-black)
3. Cropping to the bounding box of the drawn content to remove excess whitespace
4. Resizing to 28x28 and normalizing pixel values to 0-1
