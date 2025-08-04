# TagMed

## Overview
TagMed is designed to facilitate the annotation of sensitive ultrasound images and corresponding medical reports for computer vision projects. 

## Features
Medical image annotation, particularly in ultrasound, requires precision, efficiency, and a secure workflow. This tool aims to:

    Provide a simple and intuitive interface for annotating medical data.
    Allow creation and classification of bounding boxes on image frames.
    Support video annotation with automation assistance using [SAM2](https://github.com/facebookresearch/sam2) and [MedSAM2](https://github.com/bowang-lab/MedSAM2/tree/main).
    
## Quick Start

### 1. Installation

### 2. Usage

## File Structure

#### Annotation Table
When using TagMed an Annotation Table is created for you and your ImageDatabase. 


#### Your Database
```
ImageDatabase
├── Patient 1
    ├── Exam 1
        ├── Image 1
        ├── Image 2
        ├── Video 1
        ├── Frame 1 of Video 1
    ├── Exam 2
├── Patient 2
    ├── Exam 1
```
We recommend that you name or number your images as follows:
{num_patient}_{num_exam}_{num_image}

Example: 000001_01_00001

Videos should be named in the same pattern. If your videos already are framed add the frame number as suffix.
Example: 000001_01_00001_frame00001

## Citation
If you use TagMed in your research, please cite:

```
[to be added]
```

If you used the tracking feature implemented in TagMed, please also cite the corresponding publications:

SAM2
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```
MedSAM2
```
@article{MedSAM2,
    title={MedSAM2: Segment Anything in 3D Medical Images and Videos},
    author={Ma, Jun and Yang, Zongxin and Kim, Sumin and Chen, Bihui and Baharoon, Mohammed and Fallahpour, Adibvafa and Asakereh, Reza and Lyu, Hongwei and Wang, Bo},
    journal={arXiv preprint arXiv:2504.03600},
    year={2025}
}
```
EfficientTAM
```
@article{xiong2024efficienttam,
    title={Efficient Track Anything},
    author={Yunyang Xiong, Chong Zhou, Xiaoyu Xiang, Lemeng Wu, Chenchen Zhu, Zechun Liu, Saksham Suri, Balakrishnan Varadarajan, Ramya Akula, Forrest Iandola, Raghuraman Krishnamoorthi, Bilge Soran, Vikas Chandra},
    journal={preprint arXiv:2411.18933},
    year={2024}
}
```



## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Email: [Janik.Hohmann@med.uni-duesseldorf.de]
