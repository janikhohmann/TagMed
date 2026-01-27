# TagMed
<img src="resources/images/TagMed_LOGO.png"> 


<div align="center">
 <table align="center">
   <tr>
     <td><a href="https://arxiv.org" target="_blank"><img src="https://img.shields.io/badge/arXiv-Paper-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a></td>
     <td><a href="https://www.uniklinik-duesseldorf.de/patienten-besucher/klinikeninstitutezentren/klinik-fuer-gastroenterologie-hepatologie-und-infektiologie/forschung-und-lehre/arbeitsgruppen/ag-luedde" target="_blank"><img           src="https://img.shields.io/badge/UKD-AG Lüdde-f7ac00?style=for-the-badge&logoColor=white" alt="AG Lüdde"></a></td>
     <td><a href="https://github.com/janikhohmann/TagMed" target="_blank"><img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github&logoColor=white" alt="Code"></a></td>
   </tr>
 </table>
</div>


## Overview
TagMed is designed to facilitate the annotation of sensitive ultrasound images and corresponding medical reports for computer vision projects. 

<img src="resources/images/TagMed_graphical_abstract.png">
**Graphical Overview of the TagMed Annotation Framework.** **(A)** Liver ultrasound videos
from four diagnostic entities (Hepatocellular carcinoma (HCC), Metastasis (MET), Focal nodular
hyperplasia (FNH), Hemangioma (HEM)) were imported into TagMed. Every 10th frame was manually
annotated by two expert sonographers using pixel-wise segmentation masks and bounding boxes to
create the clinical ground truth. **(B)** TagMed performs automatic dataset indexing, displays optional
medical reports, and enables interactive initialization of object regions via polygons or bounding boxes.
Integrated SAM-based segmentation models (SAM2, SAM3, MedSAM2) propagate masks frame-to-
frame, allowing rapid annotation with optional corrective prompts. All annotations are processed and
stored locally to ensure data-privacy compliance. **(C)** Automatically generated segmentations are
compared with expert ground truth using Dice Similarity Coefficient (DSC) Score. Additional analyses
include temporal stability across video sequences, and differences between bounding-box and mask-
based prompts. Image of the liver (A) is adapted from Servier Medical Art (https://smart.servier.com),
licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

## Features
Medical image annotation, particularly in ultrasound, requires precision, efficiency, and a secure workflow. This tool aims to:

- Provides a simple and intuitive interface for annotating medical data with bounding boxes, polygons and class assignment.
- Allow creation and classification of bounding boxes on image frames.
- Support video annotation with frame navigation and object tracking using the automated assistance of [SAM2](https://github.com/facebookresearch/sam2), [MedSAM2](https://github.com/bowang-lab/MedSAM2) and [SAM3](https://github.com/facebookresearch/sam3).
- Designed to handle sensitive medical data locally.
    
## Quick Start

### 1. Installation

**Clone the repository:**
```bash
git clone https://github.com/janikhohmann/TagMed.git
cd TagMed
```

**Install all requirements and create an virtual environment:**
```bash
python setup_TagMed.py
```

### 2. Usage

**Activate the environment and start TagMed:**
```bash
source tagmed-env/bin/activate
cd src
python main.py
```


**Basic workflow:**
1. Load your medical image database
2. Select patient and examination
3. Choose annotation type (bounding box or polygon)
4. Annotate your images/videos
5. Use interactive segmentation models for tracking and automated propagation across video frames
6. Export annotations for your computer vision projects


## File Structure

#### Annotation Table
Bounding box annotations are saved in [x,y,h,w,]-format. Helper functions for converting to the SAM2 format and reading from the annotation table are available.
When using TagMed, an annotation table is automatically created for you and your ImageDatabase. To ensure everything works correctly, please follow the data structure outlined below.

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
We recommend that you name or number your images as follows: <br>
```
{num_patient}_{num_exam}_{num_image}
Example: 000001_01_00001.jpg
```
Videos should be named in the same pattern. If your videos are already framed, add the frame number as a suffix. If not, TagMed will automatically divide the videos into frames. These frames are not saved, but are only generated for annotation purposes.
```
{num_patient}_{num_exam}_{num_image}_frame{num_frame}
Example: 000001_01_00001_frame00001.jpg
```


#### Medical Reports
To use this function, a table containing all medical reports is required.
TagMed searches for the patient and exam identifiers in the annotation table and returns the corresponding columns from the medical reports file.
For more insights, please take a look at the available example reports.

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
SAM3
```
@misc{carion2025sam3segmentconcepts,
      title={SAM 3: Segment Anything with Concepts},
      author={Nicolas Carion and Laura Gustafson and Yuan-Ting Hu and Shoubhik Debnath and Ronghang Hu and Didac Suris and Chaitanya Ryali and Kalyan Vasudev Alwala and Haitham Khedr and Andrew Huang and Jie Lei and Tengyu Ma and Baishan Guo and Arpit Kalla and Markus Marks and Joseph Greer and Meng Wang and Peize Sun and Roman Rädle and Triantafyllos Afouras and Effrosyni Mavroudi and Katherine Xu and Tsung-Han Wu and Yu Zhou and Liliane Momeni and Rishi Hazra and Shuangrui Ding and Sagar Vaze and Francois Porcher and Feng Li and Siyuan Li and Aishwarya Kamath and Ho Kei Cheng and Piotr Dollár and Nikhila Ravi and Kate Saenko and Pengchuan Zhang and Christoph Feichtenhofer},
      year={2025},
      eprint={2511.16719},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.16719},
}
```



## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Email: Janik.Hohmann@med.uni-duesseldorf.de

## License
Please use this tool under **License [TO BE DONE]** with citing our paper.<br>
Design: Copyright (c) 2021 [rdbende](https://github.com/rdbende/Azure-ttk-theme) with MIT License. 
