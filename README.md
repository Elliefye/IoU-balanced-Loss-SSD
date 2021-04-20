# SSD (Single-Shot MultiBox Detector) with IoU balanced loss
 
A super simple SSD implementation using PyTorch, replacing SmoothL1 loss with IoU balanced loss for localization.

Based on https://github.com/uvipen/SSD-pytorch (using ResNet50 as a backbone instead of VGG-16)

Configured for a subset of OIDv6 dataset: Horse, Knife and Human body classes.
