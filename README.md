# revex_framework

**Authors:** F. Xavier Gaya-Morey, Jose M. Buades-Rubio, I. Scott MacKenzie and Cristina Manresa-Yee

<img style="width: 100%;" src="resources/graphical abstract.png"/>

This project contains the implementation of REVEX (REmoval-based Video EXplanations), a unified framework for removal-based explanations in the video domain. By decomposing explanation into manageable steps, REVEX facilitates the study of each step's impact and allow for further refinement of explanation methods. 

Additionally, six existing explanation techniques are adapted to video data by incorporating temporal information and enabling local explanations. The project also provides implementation for evaluation metrics commonly used for eXplainable AI (XAI) methods.

| Input video | Video LIME | Video K.-SHAP | Video RISE | Video LOCO | Video UP | Video SOS |
|-------------|------------|---------------|------------|------------|----------|-----------|
| ![](<resources/gifs/blowing glass small.gif>) | ![](<resources/gifs/VideoLIME.gif>) | ![](<resources/gifs/VideoKernelSHAP.gif>) | ![](<resources/gifs/VideoRISE.gif>) | ![](<resources/gifs/VideoLOCO.gif>) | ![](<resources/gifs/VideoUP.gif>) | ![](<resources/gifs/VideoSOS.gif>) |


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

## Citation

If you use this code in your research, please cite our paper:
```
@misc{gaya-morey2024revex,
	title         = {REVEX: A Unified Framework for Removal-Based Explainable Artificial Intelligence in Video},
	author        = {F. Xavier Gaya-Morey and Jose M. Buades-Rubio and I. Scott MacKenzie and Cristina Manresa-Yee},
	year          = 2024,
	url           = {https://arxiv.org/abs/2401.11796},
	eprint        = {2401.11796},
	archiveprefix = {arXiv},
	primaryclass  = {cs.CV}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.TXT) file for details.
