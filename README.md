## RAW to Tone-mapped HDR Camera ISP
<br/>

<img src="Visual comparison with built in ISPs.jpg"/>

<br/>

<img src="Visual comparison with reference models.jpg"/>

[[Paper]](https:)

The implementation of the model architecture is provided via this repository. Throughout the implementation, PyNET model Pytorch implementation in [this link] (https://github.com/aiff22/PyNET-PyTorch.git) was used as the base model which is described in [this paper] (https://arxiv.org/pdf/2002.05509.pdf). In order to construct 8 bit tone-mapped HDR image for given 10 bit Bayer pattern raw input data, a suitable dataset was created by editing subset of images in the Zurich dataset with Topaz Gigapixel AI and Photomatix Pro 5.0 software. The prepared dataset will be shared, when it is requested. 

This is the first study in which histogram information was utilized in the loss function to achieve better color constancy in the constructed images. 

<br/>

<img src="Model Architecture.png"/>

According to the NIMA, MUSIQ and LIQE no-reference image quality metric performance of our proposed model outputs, it is seen that high quality outputs can be constructed for a given Bayer pattern raw data.# raw_to_tonemappedHDR

repository provides PyTorch implementation of the RAW-to-RGB mapping approach and PyNET CNN presented in [this paper](https://arxiv.org/). The model is trained to convert **RAW Bayer data** obtained directly from mobile camera sensor into photos captured with a professional Canon 5D DSLR camera, thus replacing the entire hand-crafted ISP camera pipeline. The provided pre-trained PyNET model can be used to generate full-resolution **12MP photos** from RAW (DNG) image files captured using the Sony Exmor IMX380 camera sensor. More visual results of this approach for the Huawei P20 and BlackBerry KeyOne smartphones can be found [here](http://people.ee.ethz.ch/~ihnatova/pynet.html#demo).

Today, a conventional digital camera ISP (Image Signal Processor) pipeline is constructed by sequentially connected image processing modules which are each designed for a specific task, such as de-mosaicking, de-noising, white balance, color enhancement, tone-mapping etc. Since these modules are designed independently and optimization of the pipeline beginning-to-end is not applicable, distortions are likely to be observed in the final constructed image. To eliminate these effects, various camera ISP pipeline models have been proposed in the literature and higher image quality results have been reported utilizing convolutional neural networks. However, none of them aimed to cover HDR (High Dynamic Range) tone-mapping at the same time. In this paper, a U-Net based architecture is proposed to construct tone-mapped HDR image for given Bayer pattern raw input data. 

<br/>

#### 2. Prerequisites

- Python: scipy, numpy, imageio and pillow packages
- [PyTorch + TorchVision](https://pytorch.org/) libraries
- Nvidia GPU
