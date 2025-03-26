# VIIDA and InViDe: Computational approaches for generating and evaluating inclusive image paragraphs for the visually impaired
This repository contains the proposed methods of the paper [VIIDA and InViDe: Computational approaches for generating and evaluating inclusive image paragraphs for the visually impaired](https://www.tandfonline.com/doi/abs/10.1080/17483107.2024.2437567) published in the journal [Disability and Rehabilitation: Assistive Technology](https://www.tandfonline.com/journals/iidt20).

**Code and details about InViDe metric will be available soon!**

## Latest Updates
- **Mar-21-25:** The **VIIDA** code is released! 🚀
- **Mar-19-25:** The **VIIDA** demo is now available! Try it out [here](./viida.ipynb)! 🔥
- **Dec-11-24:** The **VIIDA** and **InViDe** paper has been published in the journal Disability and Rehabilitation: Assistive Technology! 🎊
- **Nov-22-24:** The **VIIDA** and **InViDe** paper was accepted for publication in the journal Disability and Rehabilitation: Assistive Technology! 🌟
  
## Overview
Existing image description methods when used as Assistive Technologies often fall short in meeting the needs of blind or low vision (BLV) individuals. They tend to either compress all visual elements into brief captions, create disjointed sentences for each image region, or provide extensive descriptions. To address these limitations, we introduce **VIIDA**, a procedure aimed at the Visually Impaired which implements an Image Description Approach, focusing on webinar scenes. We also propose **InViDe**, an Inclusive Visual Description metric, a novel approach for evaluating image descriptions targeting BLV people. We reviewed existing methods and developed **VIIDA** by integrating a multimodal Visual Question Answering model with Natural Language Processing (NLP) filters. A scene graph-based algorithm was then applied to structure final paragraphs. By employing NLP tools, **InViDe** conducts a multicriteria analysis based on accessibility standards and guidelines. Experiments statistically demonstrate that **VIIDA** generates descriptions closely aligned with image content as well as human-written linguistic features, and that suit BLV needs. **InViDe** offers valuable insights into the behaviour of the compared methods – among them, state-of-the-art methods based on Large Language Models – across diverse criteria. **VIIDA** and **InViDe** emerge as efficient Assistive Technologies, combining Artificial Intelligence models and computational/mathematical techniques to generate and evaluate image descriptions for the visually impaired with low computational costs. This work is anticipated to inspire further research and application development in the domain of Assistive Technologies.

## Contributions
- We present the **VIIDA**, a low-cost computational method for generating image paragraphs aligned with accessibility standards and designed to assist people with visual impairments as an Assistive Technology;
- We also propose **InViDe**, a new multicriteria metric designed to evaluate the suitability of image textual descriptions for visually impaired audiences based on accessibility standards and guidelines;
- As this work is one of the few studies in the area and is characterized by flexibility and interpretability, researchers can use the approaches presented here to produce new or improve existing Assistive Technologies for the visually impaired.

## Getting Started
### 1) Installation 

First, clone our repository.

```bash
git clone https://github.com/daniellf/VIIDA-and-InViDe.git
cd VIIDA-and-InViDe
```
Next, install the required dependencies using either **pip** or **conda**, as shown below. 

#### Option 1: Using pip  
> [!CAUTION]
> Make sure you have Python **3.9.13** or **3.9.21** installed, then run:  

```bash
pip install -r requirements.txt
```

#### Option 2: Using conda
If you prefer to use conda, you can create a new environment and install the dependencies with:

```bash
conda create --name my_env --file requirements2.txt
conda activate my_env
```
> [!NOTE]
> Replace `my_env` with your preferred environment name, e.g., `viida`.

#### **(Optional) GPU Acceleration**
> [!IMPORTANT]
> If you have a GPU and want to accelerate **VIIDA** with CUDA, you'll also need to install compatible versions of torch (`version+cu117`) and torchvision (`version+cu117`). The required version can be checked in the requirements file.

### 2) Configuration

After navigating to the `VIIDA-and-InViDe` folder, create a `webinar_dataset` folder and place the images you want to process there. If you already have another directory for this, update the `--path_dataset` argument when running the **VIIDA** script.

> [!TIP]
> In this work, we use the [webinar dataset](https://github.com/MaVILab-UFV/presenter-centric-dataset-SIBGRAPI-2023) proposed by [Ferreira et al.](https://ieeexplore.ieee.org/document/10347135), which contains images of presenters in the foreground, typically found in webinars, talk shows, and news.

We use **BLIP** as the **VQA** model, so it is required for **VIIDA** to work properly:

```bash
git clone https://github.com/salesforce/BLIP.git
```

Some modifications are required in the **BLIP** code:

1) In the `blip.py` and `blip_vqa.py` files inside `BLIP/models`, add `BLIP.` before `models.`, as shown below:

```python
from BLIP.models.vit import VisionTransformer, interpolate_pos_embed
from BLIP.models.med import BertConfig, BertModel, BertLMHeadModel

from BLIP.models.blip import create_vit, init_tokenizer, load_checkpoint
```

2) In the `blip_vqa.py` file, modify:

- Line 12 to:

```python
med_config = './BLIP/configs/med_config.json', 
```

- Replace lines 99 to 111 with:

```python
outputs = self.text_decoder.generate(input_ids=bos_ids,
                                     max_length=10,
                                     min_length=1,
                                     num_beams=num_beams,
                                     eos_token_id=self.tokenizer.sep_token_id,
                                     pad_token_id=self.tokenizer.pad_token_id,
                                     return_dict_in_generate=True,
                                     output_scores=True,
                                     **model_kwargs)
                
answers = []
probs = []

for i,output in enumerate(outputs['sequences']):
  score = outputs['sequences_scores']
  scores = outputs['scores'][i]
  ids = output[1:-1]
  prob = F.softmax(scores, dim=1).index_select(dim=1, index=ids)
  answer = self.tokenizer.decode(output, skip_special_tokens=True)
  answers.append(answer)
  probs.append(torch.sum(prob[0]))

return answers, probs
```

> [!WARNING]
> For the `--model_vqa_path` argument, you can either pass the [URL](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth) of the **BLIP VQA PyTorch model state** or download it from this [link](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth) into the `BLIP` folder and pass its local path as `--model_vqa_path`. The second option is the default in **VIIDA**. If you choose the first, make sure to update the `--model_vqa_path` argument accordingly.

## VIIDA - Step by Step
### 1) Running VIIDA on Your Images

To run **VIIDA**, use the `run_viida.py` script: 

```bash
python3 run_viida.py 
```

The following arguments can be provided: `--path_dataset`, `--img_format`, `--dense_captioning`, `--model_vqa_path`, `--ttext_sim`, `--ttext_yes`, `--dense_model`, and `--dense_file`. Here is a brief description of each argument:

- `--path_dataset`: Path to the image dataset. **Default:** `'./webinar_dataset/'`
- `--img_format`: Image file format  (e.g., `.jpg`, `.png`). **Default:** `'.jpg'`
- `--dense_captioning`: Set this to **True** if you used a dense captioning model. **Default:** `False`
- `--model_vqa_path`: Path to the BLIP-VQA `pth` file. **Default:** `'./BLIP/model_base_vqa_capfilt_large.pth'`
- `--ttext_sim`: Text similarity threshold to filter similar dense captions generated by the dense captioning model. **Default:** `0.55`
- `--ttext_yes`: Similarity threshold for validating the content described in captions using the **VQA** model. **Default:** `0.6`
- `--dense_model`: Name of the dense captioning model used (e.g., `densecap`, `grit`, or `none` if no model was used). **Default:** `'none'`
- `--dense_file`: Path to the `.json` file containing extracted dense captions. **Default:** `'./my_results.json'`

### 2) Running VIIDA-Dense or VIIDA-GRIT on Your Images

> [!CAUTION]
> [DenseCap](https://github.com/jcjohnson/densecap) and [GRIT](https://github.com/JialianW/GRiT) are independent models and are not part of our approach. Therefore, if you want to run **VIIDA** with the Visual Information Extraction stage, you need to run one of these dense captioning models first and provide the generated `.json` file to the `--dense_file` argument.

> [!WARNING]
> All images in the `webinar_dataset` folder must have been processed by the dense captioning model and included in the `.json` file.

If you choose this version of **VIIDA**, make sure to adjust the following arguments: `--dense_captioning`, `--dense_model`, and `--dense_file`, as shown in the example below:

```bash
python3 run_viida.py --dense_captioning True --dense_model 'densecap' --dense_file './results_densecap.json' 
```

> [!TIP]
> If you wish to use a dense captioning model other than those used in our paper, there is a designated field in the code to implement the reading of the `.json` file generated by your chosen model.

As a result, **VIIDA** generates an output `.json` file named `paragraphs.json`. Here is an example:

```json
[{"image_id": "gBGBkllBMyg#83640.jpg", "paragraph": "A brown woman with surprised expression. She has brown eyes and black long hair. She is wearing a white blouse, makeup and silver necklace. She is in the office, with a plant in the background."}]
```
### 3) Inference demo:
If you prefer a more interactive approach, try running our demo using [Jupyter Notebook](./viida.ipynb)!

## InViDe - Step by Step
...


## Acknowledgments
The authors would like to thank the agencies CAPES, FAPEMIG, and CNPq for funding different parts of this work.

## Contact
### Authors
- [Daniel L. Fernandes](https://github.com/daniellf) - Postdoctoral researcher - Universidade Federal de Viçosa - daniel.louzada@ufv.br
- Marcos H. F. Ribeiro - Professor - Universidade Federal de Viçosa - marcosh.ribeiro@ufv.br
- [Michel M. Silva](https://michelmelosilva.github.io/) - Assistant professor - Universidade Federal de Viçosa - michel.m.silva@ufv.br
- Fabio R. Cerqueira - Associate professor - Universidade Federal Fluminense - frcerqueira@id.uff.br

## Citation
  If you find this code useful for your research, please cite the paper:
```bibtex
@article{fernandes2024viida,
  title={VIIDA and InViDe: computational approaches for generating and evaluating inclusive image paragraphs for the visually impaired},
  author={Fernandes, Daniel L and Ribeiro, Marcos H F and Silva, Michel M and Cerqueira, Fabio R},
  journal={Disability and Rehabilitation: Assistive Technology},
  pages={1--26},
  year={2024},
  publisher={Taylor \& Francis},
  doi={10.1080/17483107.2024.2437567}
}
```
