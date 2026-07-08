# SPPrompt-Water
The code repository of this project is for the paper titled "Breaking the Scale Barrier: A Scale-Progressive Water Body Extraction Framework Inspired by Human Cognitive Mechanisms." We will make the dataset and the usage of the code public here. The abstract of the paper is as follows
![picname](https://github.com/XinMe/SPPrompt-Water/blob/main/picture/Framework.jpg?raw=true) 
**Abstract** 
Water bodies are typical natural features that exhibit significant variations in scale and appearance in high-resolution remote sensing imagery due to the influence of factors such as topography, temperature, sediment content, and aquatic vegetation. Existing deep learning-based water body segmentation networks typically adopt a local sliding-window approach for training and inference. However, this strategy limits the receptive field and hampers the model’s ability to utilize contextual information for identifying large-scale, complex-textured water bodies, often resulting in misclassification. To address these limitations, We propose a Scale-Progressive Prompt Water Body Extraction Framework (SPPrompt-Water), inspired by the hierarchical “global-to-local” cognitive mechanisms of human vision. The framework employs progressive prompting to facilitate cross-scale contextual knowledge transfer, thereby enabling high-precision water body extraction. To support this framework, we design an enforced prompt-based parallel simulation training strategy, which leverages cross-scale sample simulation and label decomposition to obtain object-level supervision within the same scene. Through parallel contrastive training, the model learns to perform segmentation with enforced prompt guidance. Extensive experiments conducted on the GID and CWBSD datasets demonstrate that the proposed method effectively balances spatial continuity in large-scale water bodies with boundary precision in small-scale ones. Our method achieves mean Intersection over Union (mIoU) scores of 83.22% and 92.27% on the respective datasets, outperforming existing approaches by 8.22% and 6.65%. These results validate the effectiveness of the proposed framework and offer a novel solution for high-precision dynamic monitoring of water bodies using remote sensing imagery.

![picname](https://github.com/XinMe/SPPrompt-Water/blob/main/picture/SPPrompt-Water%20Net.jpg?raw=true)

# CWBSD datasets
To facilitate reproducibility and further research on complex water body segmentation, we have released the official test set via **Baidu Netdisk**. Researchers can download the data using the link and access code provided below.

-   **Download link:** [https://pan.baidu.com/s/1PL6J8yHYRoNQ-xXSE1ZPEA?pwd=xftt]
    
-   **Access code:** [xftt]

# Code
## ⚙️ Environment Setup
###  Requirements
 Recommend version
-   Python >= 3.10
-   PyTorch >= 2.1
Other dependencies can be installed according to the `requirements.txt` file provided in the repository.


## 📊 Dataset Preparation

To train and evaluate the proposed scale-progressive framework, please prepare the dataset as follows:

### 1️⃣ Dataset Split

-   Prepare any remote sensing segmentation dataset.
    
-   Split the dataset into **training**, **validation**, and **test** sets.
    
-   The ground-truth masks should be binary with pixel values:
    
    -   `0` for background
        
    -   `255` for water
        

----------

### 2️⃣ Multi-Scale Generation (Different Levels)

For scale-progressive training, generate multiple resolution levels:

-   Resize each full-scene image **together with its corresponding label** to different resolutions, e.g.:
    
    -   2× downsample
        
    -   4× downsample
        
    -   8× downsample
        
-   Each resized version is treated as one **scale level** (e.g., `level0`, `level1`, ...).
    

----------

### 3️⃣ Patch Cropping

For each scale level:

-   Crop both images and labels into **1024 × 1024** patches.
    
-   Overlapping cropping is optional depending on your setting.
    

----------

### 4️⃣ Prompt Mask Generation

For each label patch:

-   Apply morphological operations (e.g., erosion and dilation) to construct prompt guidance masks.
    
-   Downsample the processed label by **4×** to generate a **256 × 256 prompt mask**.
    
-   Store it under the folder `prompt_mask_256`.
    

----------

### 5️⃣ Directory Structure

Organize the dataset in the following format:

data/  
├── level0/  
│ ├── train/  
│ │ ├── imgs/  
│ │ ├── gts/  
│ │ └── prompt_mask_256/  
│ ├── val/  
│ │ ├── imgs/  
│ │ ├── gts/  
│ │ └── prompt_mask_256/  
│  
├── level1/  
│ ├── train/  
│ │ ├── imgs/  
│ │ ├── gts/  
│ │ └── prompt_mask_256/  
│ ├── val/  
│ │ ├── imgs/  
│ │ ├── gts/  
│ │ └── prompt_mask_256/  
│  
└── ...

Modify the dataset path in the file.

## 🚀 Training

python train_promptwaternet.py


----------

## 🔍 Inference

### Standard Sliding-Window Inference

python inference_waterprompt.py

### Scale-Progressive Prompt-Guided Inference (Proposed)

python PyramidPromptSeg_mutilayer.py

**Note:** We recommend modifying the default arguments directly in the corresponding files if needed (e.g., dataset paths, batch size, number of levels). You can adjust the default values defined in `argparse` for convenience.

