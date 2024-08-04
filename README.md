## RAW to Tone-mapped HDR Camera ISP

<br/>

<img src="Visual comparison with built in ISPs.jpg"/>

<br/>

<img src="Visual comparison with reference models.jpg"/>

[[Paper]](https:)
#### 1. Overview
The implementation of the model architecture is provided via this repository. Throughout the implementation, PyNET model Pytorch implementation in [this link] (https://github.com/aiff22/PyNET-PyTorch.git) was used as the base model which is described in [this paper] (https://arxiv.org/pdf/2002.05509.pdf). In order to construct 8 bit tone-mapped HDR image for given 10 bit Bayer pattern raw input data, a suitable dataset was created by editing subset of images in the Zurich dataset with Topaz Gigapixel AI and Photomatix Pro 5.0 software. The prepared dataset will be shared, when it is requested. 

This is the first study in which histogram information was utilized in the loss function to achieve better color constancy in the constructed images. The methodology defined in [HistoGAN] (https://arxiv.org/abs/2011.11731) and implementation described in [this link] (https://github.com/mahmoudnafifi/HistoGAN.git) was used for extracting RGB-uv features. Then Hellinger Distance function between the differentiable histograms computed from the groundtruth and predicted images. 

#### 2. Model Architecture
<br/>

<img src="Model Architecture.png"/>

The proposed model architecture is presented above. Our model consists of five layers, and each layer needs to be trained separately starting from the bottom layer. The spatial feature resolution is reduced at Layer-2, Layer-3 and Layer-4 with Max Pooling layer by additional factor of two in each layer. Layer-1 is trained in the same spatial feature resolution with the input data, which is 224 x 224. Finally in the output of Level-0 ground-truth feature resolution, 448 x 448, is obtained by upscaling the output of the Layer-1.

According to the NIMA, MUSIQ and LIQE no-reference image quality metrics, it is seen that high quality outputs can be constructed for a given Bayer pattern raw data with our proposed methodology.# raw_to_tonemappedHDR

#### 3. Training Details
<br/>

<img src="Training Details.jpg"/>

The model is trained with the batch sizes and total number of epochs presented above. ADAM optimization is used optimize nearly 6.5M parameters in the model architecture. Our model is trained on NVIDIA RTX A5000 GPU. 
- Python: scipy, numpy, imageio and pillow packages
- [PyTorch + TorchVision](https://pytorch.org/) libraries
- Nvidia GPU
