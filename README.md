# Bean Leaf Disease Classification

## Task
Supervised image classification of bean leaf images into:
- angular_leaf_spot
- bean_rust
- healthy

## Model
EfficientNet-B0 with transfer learning

## Dataset
Beans dataset: https://github.com/AI-Lab-Makerere/ibean/?tab=readme-ov-file
- Train: 1034
- Validation: 133
- Test: 128

## Results
- Best validation macro F1: 0.9925
- Test accuracy: 0.9766
- Test macro F1: 0.9767

## Run
python train.py
