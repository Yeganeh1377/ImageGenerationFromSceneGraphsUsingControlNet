# Image Generation Conditioned on Scene Graphs

Using text as conditioning in image generation models has some shortcomings. Image descriptions may be long, ambiguous, and loosely structured. Scene graphs could be a more structured alternative for representing image contents. In this project I investigated the performance of ControlNet conditioned on scene graphs with the current benchmark, [sg2im](https://github.com/google/sg2im) on Visual Genome Dataset. An overview of the project can be seen below:

![alt text](https://github.com/Yeganeh1377/ImageGenerationFromSceneGraphsUsingControlNet/blob/main/images/fig1.jpg)


## Setup
The folder src contains the script for preprocessing the data for ControlNet and the scripts used for inference. The pretrained models can be optained from [sg2im](https://github.com/google/sg2im) and [HuggingFace]([https://github.com/google/sg2im](https://huggingface.co/docs/diffusers/training/controlnet)https://huggingface.co/docs/diffusers/training/controlnet). I used ControlNet v1.1 and Stable Diffusion v1.5. The preprocessing of the data is identical to sg2im. You can some of the examples [here] (https://github.com/Yeganeh1377/ImageGenerationFromSceneGraphsUsingControlNet/tree/main/images).
