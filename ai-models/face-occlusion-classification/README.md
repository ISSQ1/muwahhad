# Face Occlusion Classification 🕶️

This project focuses on detecting **face occlusions** in uploaded ID photos — such as masks, sunglasses, or any obstruction — to ensure the image meets official identification standards.

If the face is covered or partially hidden, the model automatically flags the image as **Occluded face**.

## 🧠 Model Overview
The model is trained to classify whether a person’s face is:
- **Clear (compliant)**  
- **Occluded (non-compliant)**

It uses a fine-tuned **ResNet18** architecture with custom preprocessing and augmentation for real-world accuracy.
