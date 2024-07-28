# Autonomous Mars Rover: Terrain-Aware Navigation
![Hero](https://github.com/AnshChoudhary/Autonomous-Mars-Rover/blob/main/Mars-Rover-Hero.jpg)
## Overview
The Autonomous Mars Rover project aims to revolutionize terrain-aware navigation for rovers on Mars by leveraging advanced machine learning techniques. Utilizing the U-Net architecture, our model is designed to perform semantic segmentation of Martian landscapes, identifying various terrain types that are critical for safe and efficient rover navigation. This project has the potential to be groundbreaking in the field of space exploration, enhancing the autonomy and decision-making capabilities of rovers as they traverse the challenging Martian environment.

## Project Motivation
Navigating the Martian surface is fraught with challenges, from avoiding hazardous terrain to selecting the most efficient paths for exploration. Traditional methods of navigation rely heavily on pre-mapped routes and human intervention, limiting the scope and speed of exploration. Our project seeks to enhance rover autonomy by providing real-time terrain segmentation and analysis, allowing rovers to make informed decisions independently.

## Key Features
- U-Net Architecture: Our model employs a U-Net architecture, renowned for its efficacy in image segmentation tasks. This enables precise identification of terrain features such as soil, bedrock, sand, and large rocks.

- Crowdsourced vs. Predicted Segmentation: The project demonstrates significant improvements in segmentation accuracy by comparing crowdsourced data against our model's predictions.

- AI4Mars Dataset: We utilize the AI4Mars dataset, which comprises expertly labeled Martian terrain images. This dataset is essential for training and validating our segmentation model.

# Dataset Information
The dataset used in this project is a version of the AI4Mars merged dataset:

### Sources: 
The dataset contains images from the Curiosity Rover (MSL), with planned inclusion of Mars Exploration Rover (MER) data.

### Labels:

- Soil: (0,0,0)
- Bedrock: (1,1,1)
- Sand: (2,2,2)
- Big Rock: (3,3,3)
- NULL: (255,255,255)

### Training Data: 
Crowdsourced labels with a minimum agreement of 3 labelers and 2/3 agreement for each pixel. Distances beyond 30 meters are masked, as is the rover itself.

### Test Data: 
Expert-gathered labels with 100% pixel agreement required. Different versions specify varying levels of consensus among labelers.

## Image Information
- EDR (Engineering Data Record): The actual Martian images used for training and inference.

- Rover Masks (mxy): Binary masks used to exclude the rover from the segmentation process.

- Range Masks (rng-30m): Binary masks that exclude terrain beyond 30 meters, enhancing the relevance of the segmentation.

## Training and Evaluation
The project's core components include two scripts:

- train.py: This script trains the U-Net model on the AI4Mars dataset, optimizing for accuracy and Intersection over Union (IoU).

![](https://github.com/AnshChoudhary/Autonomous-Mars-Rover/blob/main/static/iou_acc.png)

- train-eval.py: This script evaluates the trained model's performance on the test set, providing visual comparisons between crowdsourced and predicted segmentations.

## Results
Our model has achieved the following performance metrics on the AI4Mars dataset:

- Accuracy: 0.8219
- IoU: 0.7004

The results demonstrate the model's ability to effectively segment Martian terrain, distinguishing between different terrain types with high accuracy. The image below illustrates the comparison between crowdsourced and predicted segmentations:

![Segmentation of Mars Terrain](https://github.com/AnshChoudhary/Autonomous-Mars-Rover/blob/main/static/segmentation.png)

## Groundbreaking Implications
The Autonomous Mars Rover project represents a significant leap forward in space exploration technology. By enabling rovers to autonomously interpret and respond to their environments, we can extend the range and duration of missions, reduce the reliance on Earth-based instructions, and increase the scientific yield of each mission. This project lays the foundation for future exploration endeavors, where autonomous systems will play a crucial role in unraveling the mysteries of Mars and beyond.

## Future Work
- Dataset Expansion: Incorporate additional datasets, including MER data, to enhance model robustness and generalizability.

- Model Optimization: Experiment with different architectures and hyperparameters to further improve segmentation accuracy and efficiency.

- Real-time Deployment: Develop systems for deploying the model in real-time on rover hardware, facilitating on-the-fly decision-making during missions.

## Getting Started

Prerequisites: 
- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## Installation
Clone the repository:

```bash
git clone https://github.com/AnshChoudhary/Autonomous-Mars-Rover.git
cd Autonomous-Mars-Rover
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Running the Training Script
To train the model on the AI4Mars dataset, execute:

```bash
python train.py
```

## Running the Evaluation Script
To evaluate the model's performance and visualize segmentation results, execute:

```bash
python train-eval.py
```

## Contributing
We welcome contributions from the community! If you're interested in improving the model, expanding the dataset, or adding new features, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments
We extend our gratitude to NASA and the AI4Mars project for providing the invaluable dataset used in this research. Their work has been instrumental in advancing our understanding of Martian terrains and enabling projects like ours.
