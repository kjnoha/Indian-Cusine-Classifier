# Indian Cuisine Classifier

A deep learning image classification project that identifies different Indian food dishes from images. The model is built using a DenseNet-based architecture and trained on a curated/filtered dataset of Indian cuisine images.

## Project Overview

This project trains a convolutional neural network (DenseNet) to classify images of Indian dishes into their respective categories. It includes scripts for downloading/preparing data, training the model, evaluating accuracy, and serving predictions through a simple app.

## Project Structure

```
├── filtered_dataset/                     # Curated dataset used for training/testing
├── models/                                # Saved model artifacts
├── temp_downloads/                        # Temporary storage for downloaded data
├── app.py                                  # Application script to run predictions
├── download_data.py                        # Script to fetch/prepare the dataset
├── model6_densenet.py                       # DenseNet model training script
├── model6_densenet.h5                       # Trained DenseNet model weights
├── test_model.py                            # Script to test/evaluate the trained model
├── densenet_training_plots.html              # Training performance visualization
├── densenet_accuracy_plot.html                # Accuracy visualization
├── indian_food_classifier_plots.svg            # Summary plots
├── accuracy_calculation_diagram.svg              # Diagram explaining accuracy calculation
├── model6_results.png                             # Sample model output/results
└── 2nd_result.png                                   # Additional sample result
```

## How It Works

1. **Data preparation** — `download_data.py` collects and organizes the Indian food image dataset.
2. **Model training** — `model6_densenet.py` trains a DenseNet-based CNN on the dataset.
3. **Evaluation** — `test_model.py` evaluates the trained model's accuracy, with results visualized in the included plots.
4. **Inference** — `app.py` loads the trained model (`model6_densenet.h5`) and runs predictions on new images.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/kjnoha/Indian-Cusine-Classifier.git
   cd Indian-Cusine-Classifier
   ```
2. Install dependencies (TensorFlow/Keras, NumPy, Pandas, etc. — add a `requirements.txt` if not already present):
   ```bash
   pip install tensorflow numpy pandas matplotlib
   ```
3. Run the data download script (if needed):
   ```bash
   python download_data.py
   ```
4. Train or test the model:
   ```bash
   python model6_densenet.py      # train
   python test_model.py           # evaluate
   ```
5. Run the app to classify new images:
   ```bash
   python app.py
   ```

## Results

Training accuracy and performance plots are available in:
- `densenet_training_plots.html`
- `densenet_accuracy_plot.html`
- `indian_food_classifier_plots.svg`

Sample classification outputs are shown in `model6_results.png` and `2nd_result.png`.

## Tech Stack

- **Python**
- **DenseNet (CNN architecture)**
- **HTML** (for visualization outputs)

## Author

**kjnoha**
GitHub: [github.com/kjnoha](https://github.com/kjnoha)
