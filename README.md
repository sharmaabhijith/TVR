# Total Variation based image Resurfacing (TVR)

By [Abhijith Sharma](https://www.linkedin.com/in/abhijith-sharma/), [Phil Munz](https://www.linkedin.com/in/philmunz/), [Apurva Narayan](https://scholar.google.com/citations?user=e5OCZ1cAAAAJ&hl=en&authuser=2)

## Framework
Code for "[Assist Is Just as Important as the Goal: Image Resurfacing to Aid Model’s Robust Prediction]()" submitted in WACV 2024. 

<img src="./Figures/TVD.PNG" width="700" height="280" /> 

**Takeaways**: 
1. Adversarial patches threaten visual AI models in the real world.
2. Number of patches in an attack is variable and determines its potency in a specific environment.
3. Existing defenses assume a single patch in the scene, and the multi-patch scenario is shown to overcome them.
4. The TVR is a model-agnostic defense against single and multi-patch attacks.
5. The TVR is an image-cleansing method that processes images to remove probable adversarial regions.
6. Nullifies the influence of patches in a single image scan with no prior assumption on the number of patches. 

### :chart_with_upwards_trend: Demo

<img src="./Figures/demo.PNG" width="630" height="280" /> 

## :page_with_curl: Requirements

Experiments were done with PyTorch 1.7.0 and timm 0.4.12. The complete list of required packages are available in `requirement.txt`, and can be installed with `pip install -r requirement.txt`. The code should be compatible with newer versions of packages. Update 04/2023: tested with `torch==1.13.1` and `timm=0.6.13`; the code should work fine.

## :open_file_folder: Files

```shell
├── README.md                        #this file 
├── requirement.text                 #required package
|
├── helper.py                        #this file 
├── image_resurfacer.py              #required package
├── main.py                          #this file 
|
├── Figures                          #this file 
├── GAN_Model                        #required package
├── GAN_Train                        #example command to run the code
|   ├── mask_data                    #utils for constructing models and data loaders
|   ├── real_data                    #utils for PatchCleanser defenses
|   ├── GAN.py                       #utils for constructing models and data loaders
|   ├── inpaint.py                   #utils for PatchCleanser defenses
|   └── mask_datagen.py              #utils for masked model training                        
| 
├── assets                           #PatchCleanser: certify robustness via two-mask correctness 
├── multi_patch                      #PatchCleanser: evaluate clean accuracy and per-example inference time
|   ├── ResNet18                     #utils for PatchCleanser defenses
|   └── VGG16                        #utils for masked model training
|
├── results
├── single_patch                     #undefended vanilla models: evaluate clean accuracy and per-example inference time
|   ├── ResNet18                     #utils for PatchCleanser defenses
|   └── VGG16                        #utils for masked model training
|
├── transforms                       #train undefended vanilla models for different datasets
|   ├── apply_patch.py               #utils for PatchCleanser defenses
|   └── my_random_affine.py          #utils for masked model training
|
└── utils                            #directory for checkpoints
    ├── utils.py                     #utils for constructing models and data loaders
    └── visualization.py             #utils for masked model training
```

## :open_book: Datasets

- [ImageNet](https://image-net.org/download.php) (ILSVRC2012)
- [ImageNette](https://github.com/fastai/imagenette) ([Full size](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz))
- [ImageNet-Patch BenchMark Dataset](https://github.com/pralab/ImageNet-Patch)
    - (The benchmark adversarial patches are already imported so you do not have to do anything)

## :newspaper: Citations

If you find our work useful in your research, please consider citing:

```tex
@inproceedings{sharma2023vulnerability,
  title={Vulnerability of CNNs against Multi-Patch Attacks},
  author={Sharma, Abhijith and Bian, Yijun and Nanda, Vatsal and Munz, Phil and Narayan, Apurva},
  booktitle={Proceedings of the 2023 ACM Workshop on Secure and Trustworthy Cyber-Physical Systems},
  pages={23--32},
  year={2023}
}
```
