RAW to Tone-mapped HDR Camera ISP
Today, a conventional digital camera ISP (Image Signal Processor) pipeline is constructed by sequentially connected image processing modules which are each designed for a specific task, such as de-mosaicking, de-noising, white balance, color enhancement, tone-mapping etc. Since these modules are designed independently and optimization of the pipeline beginning-to-end is not applicable, distortions are likely to be observed in the final constructed image. To eliminate these effects, various camera ISP pipeline models have been proposed in the literature and higher image quality results have been reported utilizing convolutional neural networks. However, none of them aimed to cover HDR (High Dynamic Range) tone-mapping at the same time. In this paper, a U-Net based architecture is proposed to construct tone-mapped HDR image for given Bayer pattern raw input data. For this purpose a suitable dataset was created by editing subset of images in the Zurich dataset with Topaz Gigapixel AI and Photomatix Pro 5.0 software. This is the first study in which histogram information was utilized in the loss function to achieve better color constancy in the constructed images. According to the NIMA, MUSIQ and LIQE no-reference image quality metric performance of our proposed model outputs, it is seen that high quality outputs can be constructed for a given Bayer pattern raw data.# raw_to_tonemappedHDR
