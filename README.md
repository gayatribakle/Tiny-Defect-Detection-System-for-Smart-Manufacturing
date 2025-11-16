# Tiny-Defect-Detection-System-for-Smart-Manufacturing
A lightweight image classification pipeline to detect surface defects using TensorFlow with transfer learning, pruning, and quantization — optimized for resource-constrained environments.

It leverages **TensorFlow** for image classification on a custom defect dataset, applying techniques such as:

- Transfer learning with MobileNetV2 for improved accuracy
- Model pruning to reduce model size and computation
- Post-training quantization for further compression

The final output is a compressed TensorFlow Lite model optimized for edge devices.

---


## Dataset

We use the **NEU Surface Defect Dataset (NEU-DET)**, which contains images of six types of defects:

- Crazing  
- Inclusion  
- Patches  
- Pitted Surface  
- Rolled-in Scale  
- Scratches

### Link : https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

The dataset is organized into training and validation folders, each containing subfolders for each defect class.

---

## Features

- Custom TensorFlow data pipeline for loading and preprocessing images
- CNN baseline model for defect classification
- Transfer learning using MobileNetV2 for enhanced performance
- Model compression via pruning and quantization using TensorFlow Model Optimization Toolkit
- Export to TensorFlow Lite for deployment on edge/IoT devices

---

## Installation

Ensure you have Python 3.7+ installed. Install required packages:

```
pip install tensorflow matplotlib tensorflow-model-optimization
```

### Usage

Organize your dataset as:

```
defect_dataset_1/
    NEU-DET/
        train/
            images/
                crazing/
                inclusion/
                patches/
                pitted_surface/
                rolled-in_scale/
                scratches/
        validation/
            images/
                crazing/
                inclusion/
                patches/
                pitted_surface/
                rolled-in_scale/
                scratches/
```
- Run the training scripts
Train the baseline CNN or transfer learning model.

- Apply pruning and quantization
Compress the model for Tiny AI deployment.

- Export TFLite model
Use the quantized model for inference on edge devices.

Example Code Snippets

# Load dataset
```
train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, ...)
```

# Build and compile model
```
model = build_model(num_classes)
model.compile(...)
```

# Train model
```
model.fit(train_ds, validation_data=val_ds, epochs=10)
```

# Apply pruning and quantization
```
pruned_model = apply_pruning(model)
quantized_model = convert_to_tflite(pruned_model)
```
### Results


Achieved up to 85% validation accuracy with transfer learning.

Reduced model size by 70% to 85% via pruning and quantization.

Real-time inference capable on edge devices with limited compute.
<img width="1128" height="514" alt="Screenshot 2025-11-16 123045" src="https://github.com/user-attachments/assets/c48637cc-a3ed-483a-b6ac-4a79688bdeb4" />

## Output 

<img width="656" height="824" alt="Screenshot 2025-11-16 123340" src="https://github.com/user-attachments/assets/26e132d6-0787-4ec3-813f-6757897e4b52" />


