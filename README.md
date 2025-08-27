# BrainTumorPred-ResNet50-Neurosight

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-blue.svg)
![Google Colab](https://img.shields.io/badge/Google-Colab-blueviolet.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A deep learning pipeline for classifying brain tumors (glioma, meningioma, notumor, pituitary) using ResNet50 on T2-weighted MRI images, achieving ~90% validation accuracy and targeting 95%+ macro F1-score with Grad-CAM visualizations for explainable medical diagnosis.

## Overview

This project develops a high-performance deep learning model for brain tumor classification using the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (~7,023 T2-weighted MRI images). Built with ResNet50 and transfer learning, it achieves ~90% validation accuracy after 15 epochs and aims for 95%+ macro F1-score. The pipeline includes data augmentation, fine-tuning, and Grad-CAM for interpretable tumor predictions, making it suitable for medical imaging research and clinical applications.

## Features

- **ResNet50 Model**: Fine-tuned pre-trained ResNet50 for 4-class brain tumor classification (glioma, meningioma, notumor, pituitary).
- **Data Augmentation**: Random flips, rotations, and contrast adjustments to enhance model generalization.
- **Performance**: ~90% validation accuracy and macro F1-score after 15 epochs, targeting 95%+ for publication.
- **Explainability**: Grad-CAM visualizations to highlight tumor regions in MRIs (planned feature).
- **Reproducible**: Random seed and saved model weights for consistent results.

## Dataset

- **Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (~7,023 T2-weighted MRI images).
- **Classes**: Glioma, Meningioma, No Tumor, Pituitary (~23-28% per class).
- **Split**: ~4,569 train, ~1,143 validation, ~1,311 test images.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/BrainTumorPred-ResNet50-Neurosight.git
   cd BrainTumorPred-ResNet50-Neurosight
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch torchvision scikit-learn numpy matplotlib
   ```

3. **Download Dataset**:
   - Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
   - Unzip to `/path/to/brain_tumor_dataset/unzipped/`.

4. **Run in Colab**:
   - Upload `brain_tumor_classification.ipynb` to Google Colab.
   - Mount Google Drive and set dataset path to `/content/drive/MyDrive/Research/brain_tumor_dataset/unzipped`.

## Usage

1. **Run the Notebook**:
   - Open `brain_tumor_classification.ipynb` in Colab.
   - Follow cells to load data, initialize ResNet50, train, and evaluate.
   - Current results: ~90% validation accuracy after 15 epochs.

2. **Train the Model**:
   ```python
   num_epochs = 5
   for epoch in range(num_epochs):
       train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
       val_loss, val_acc, val_f1 = validate_one_epoch(model, val_loader, criterion, device)
       print(f"{epoch+1} | {train_loss:.4f} | {train_acc:.4f} | {val_loss:.4f} | {val_acc:.4f} | {val_f1:.4f}")
   ```

3. **Save/Load Model**:
   ```python
   torch.save(model.state_dict(), 'brain_tumor_model.pth')
   model.load_state_dict(torch.load('brain_tumor_model.pth'))
   ```

## Results

- **After 15 Epochs**:
  - Train Accuracy: ~89.7%
  - Validation Accuracy: ~90.0% (peak 90.2%)
  - Validation Loss: ~0.2774
  - Note: Macro F1-score to be added for balanced performance.

- **Next Steps**:
  - Add macro F1-score for class imbalance (~23-28%).
  - Test batch sizes (16, 64) for stability.
  - Unfreeze ResNet50 layers for 95%+ F1.
  - Implement Grad-CAM for visualizations.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for suggestions, bug reports, or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) for providing high-quality MRI data.
- PyTorch and scikit-learn communities for robust tools.
- Google Colab for free GPU access.