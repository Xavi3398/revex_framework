# revex_framework

**Authors:** F. Xavier Gaya-Morey, Jose M. Buades-Rubio, I. Scott MacKenzie and Cristina Manresa-Yee

This repository contains the complete code used in the scientific article titled [*"REVEX: A Unified Framework for Removal-Based Explainable Artificial Intelligence in Video"*](https://arxiv.org/abs/2401.11796). 

<img style="width: 100%;" src="resources/graphical abstract.png"/>

It contains the implementation of the REVEX (REmoval-based Video EXplanations) framework. Additionally, the project contains the implementation of six explanation techniques are adapted to video data by incorporating temporal information and enabling local explanations. The project also provides implementation for evaluation metrics commonly used for eXplainable AI (XAI) methods.

| Input video | Video LIME | Video K.-SHAP | Video RISE | Video LOCO | Video UP | Video SOS |
|-------------|------------|---------------|------------|------------|----------|-----------|
| ![](<resources/gifs/blowing glass small.gif>) | ![](<resources/gifs/VideoLIME.gif>) | ![](<resources/gifs/VideoKernelSHAP.gif>) | ![](<resources/gifs/VideoRISE.gif>) | ![](<resources/gifs/VideoLOCO.gif>) | ![](<resources/gifs/VideoUP.gif>) | ![](<resources/gifs/VideoSOS.gif>) |

## Abstract

We developed REVEX, a removal-based video explanations framework. This work extends fine-grained explanation frameworks for computer vision data and adapts six existing techniques to video by adding temporal information and local explanations. The adapted methods were evaluated across networks, datasets, image classes, and evaluation metrics. By decomposing explanation into steps, strengths and weaknesses were revealed in the studied methods, for example, on pixel clustering and perturbations in the input. Video LIME outperformed other methods with deletion values up to 31% lower and insertion up to 30% higher, depending on method and network. Video RISE achieved superior performance in the average drop metric, with values 10% lower. In contrast, localization-based metrics revealed low performance across all methods, with significant variation depending on network. Pointing game accuracy reached 53%, and IoU-based metrics remained below 20%. Drawing on the findings across XAI methods, we further examine the limitations of the employed XAI evaluation metrics and highlight their suitability in different applications.

## Install

To install the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/Xavi3398/revex_framework.git
    ```

2. Navigate to the project directory:
    ```sh
    cd revex_framework
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Install the revex_framework:
    ```sh
    pip install -e .
    ```

## Usage

[basic_usage.ipynb](notebooks/basic_usage.ipynb) offers a brief introduction on how to use the REVEX frameworks to obtain removal-based explanations. You will also find several examples featuring multiple options for the different steps.

The REVEX explanation pipeline consists of four main components:

* [segmenters.py](revex_framework/segmenters.py): Contains various options for video segmentation.
* [perturbers.py](revex_framework/perturbers.py): Provides different methods for perturbing video regions, including feature selection, sample selection, and feature removal. Each perturbation is passed through the network to obtain an associated prediction.
* [explainers.py](revex_framework/explainers.py): Summarizes the data generated from the perturbation step.
* [visualizers.py](revex_framework/visualizers.py): Offers multiple visualization options to enhance the understanding of the explanations.

In the [notebooks](notebooks) directory, you will find examples demonstrating how to use the different options available at each step.

## Acknowledgements

This work is part of the Project PID2023-149079OB-I00 (EXPLAINME) funded by MICIU/AEI/10.13039/ 501100011033/ and FEDER, EU and of Project PID2022-136779OB-C32 (PLEISAR) funded by MICIU/AEI/10.13039 /501100011033/ and FEDER, EU. 

F. X. Gaya-Morey was supported by an FPU scholarship from the Ministry of European Funds, University and Culture of the Government of the Balearic Islands.

## Citing

If you use this code in your research, please cite our paper:
```
@misc{gaya-morey2024revex,
	title         = {REVEX: A Unified Framework for Removal-Based Explainable Artificial Intelligence in Video},
	author        = {F. Xavier Gaya-Morey and Jose M. Buades-Rubio and I. Scott MacKenzie and Cristina Manresa-Yee},
	year          = 2024,
        doi           = {10.48550/arXiv.2401.11796},
	url           = {https://arxiv.org/abs/2401.11796},
	eprint        = {2401.11796}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.TXT) file for details.
