# Reproducing RealKcat Results

Welcome to the RealKcat reproduction guide! Here, you'll find everything you need to recreate our enzyme kinetics predictions for \( k_{\text{cat}} \) and \( K_M \) values. Let’s get started!

---

### 1. 📂 Download the Datasets
To begin, go to [https://chowdhurylab.github.io/downloads.html](https://chowdhurylab.github.io/downloads.html) and navigate to **KinHub-1.0** (Manually-curated Enzyme Parameter Database; verified from 2158 papers) to download all necessary files for training and inference.

---

### 2. 🚀 Training the Model
With the datasets in place, you’re ready to build your RealKcat predictor. We’ve split the training process for both \( k_{\text{cat}} \) and \( K_M \) predictions:

- **For \( k_{\text{cat}} \) Predictions**: Use the `RealKcat_kcat_train` script to train an XGBoost model specifically for accurate \( k_{\text{cat}} \) predictions.
- **For \( K_M \) Predictions**: Use the `RealKcat_km_train` script. It shares the same model architecture but is optimized to predict \( K_M \) values, providing a consistent framework for both kinetic parameters.

---


### 3. 🎉 Download Pre-trained Models
Prefer to dive straight into predictions? Download our pre-trained models here: [Download RealKcat Models](https://chowdhurylab.github.io/downloads.html). This allows you to skip the training phase and go directly to inference.

---
### 4. 🔍 Running Inference Like a Pro
Our **inference code** enables you to easily input sequences and substrates for prediction. You can input:
- **A single sequence and substrate** for a quick, focused prediction.
- **A batch of sequences and substrates** for high-throughput predictions.

Whether analyzing a single enzyme or an entire library, our inference script handles it all. Just plug in your data and let RealKcat generate precise predictions.

---

By following these steps, you’ll be able to reproduce RealKcat's robust predictions for enzyme kinetics, providing insights into both \( k_{\text{cat}} \) and \( K_M \) across diverse enzyme-substrate interactions. Happy modeling!
