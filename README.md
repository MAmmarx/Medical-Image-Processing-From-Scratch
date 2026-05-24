# Medical Image Processing and Feature Engineering From Scratch

A raw, pixel-level computer vision and data engineering repository developed in **Python** using low-level matrix mathematics. This project implements a robust medical image preprocessing pipeline designed to optimize raw MRI scans by stripping out ambient canvas space. Rather than relying on wrapped libraries, every spatial filter, morphology dilation step, and component contour bounding tracker is coded explicitly from the ground up using fundamental numerical algorithms.

**Image Engineering & Preprocessing Framework | 3rd Year University Project**

> This system serves as the foundational data engineering layer supporting advanced deep learning pipelines, demonstrating an early-career mastery of lower-level pixel manipulation, discrete math convolutions, and computer vision theory.

## Technical Deep-Dive & Low-Level Algorithms

Standard computer vision workflows hide data transformations behind simple abstractions. This framework isolates and exposes those exact spatial mechanics by replacing them with bare mathematical loops:

### 1. Manual 2D Gaussian Filtering
Instead of invoking automated blurring steps, spatial smoothing is calculated pixel-by-pixel by performing a 2D matrix convolution using a discrete kernel. The image matrix array types undergo uniform floating-point scaling before being filtered:

$$Kernel = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$

### 2. Explicit Morphological Mathematics
- **Erosion Minimum Filters:** Implements a localized neighborhood pass that evaluates a $3 \times 3$ grid around each target pixel coordinate. Out-of-bounds padding elements are locked at max value to preserve clean edge scaling, setting the target pixel to the minimum localized intensity value.
- **Dilation Maximum Filters:** Re-scans binary layouts using a structural maximum window filter, mapping values to the neighborhood extreme to seal spatial fragmentation lines within dense tissue clusters.

### 3. Graph-Based Bounding Trackers (BFS Contour Finder)
To isolate anatomical bounds without using automated contour functions, the engine models the image as an unweighted matrix graph:
- Pixels meeting intensity thresholds trigger an explicit **Breadth-First Search (BFS)** algorithm using coordinate tracking queues.
- The algorithm checks **8-connected neighbor states** around active locations to discover continuous components.
- Live tracking variables track extreme point thresholds ($Min_Y, Max_Y, Min_X, Max_X$), returning optimal crop bounds while filtering out secondary background noise channels.

## Key Features
- **Pure Matrix Implementations:** Strips out typical dependencies to perform image transformations directly inside standard multi-dimensional NumPy arrays.
- **Granular Neighborhood Testing:** Employs precise window kernels to calculate custom smoothing distributions and binary threshold boundaries manually.
- **8-Directional Graph Walking:** Virtualizes discrete component structures using optimized coordinate loops to identify the largest region of interest (ROI).
- **Destructive Noise Suppression:** Merges consecutive erosion and dilation steps to clear out background artifacting without affecting structural tissue boundaries.
- **Deep-Learning Ready Outputs:** Seamlessly integrates into PyTorch transformation loops, formatting raw images into normalized, high-contrast, cropped tensors.

## Project Structure
```text
├── dataset/
│   ├── Training/             # Input raw multi-class target arrays 
│   └── Testing/              # Unprocessed validation image streams
└── Final_Image_Processing_Project.ipynb # Complete, self-contained interactive development notebook
