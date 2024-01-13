# Preserving Privacy Through Dememorization: An Unlearning Technique For Mitigating Memorization Risks In Language Models

This is the official repository for "[Preserving Privacy Through Dememorization: An Unlearning Technique For Mitigating Memorization Risks In Language Models](https://aclanthology.org/2023.emnlp-main.265/)"

## Table of Contents

- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Questions](#questions)
- [DeMemorized Model Weights](#dememorized-model-weights)
- [Citation](#citation)

## Installation
To install the code's requirements, do the following:
```
pip install -r requirements.txt
```

## Training
For training to DeMemorize, run the following notebook:

Run the following [notebook](https://github.com/Alymostafa/DeMemorization/blob/main/code/train/train.ipynb)

## Evaluation
After you have a DeMemorized model, you can evaluate it using this [script](https://github.com/Alymostafa/DeMemorization/blob/main/code/eval/dememorize_eval.py)

## Questions
If you have a question, don't hesitate to open an issue or reach out to Aly Kassem @ kassem6@uwindsor.ca


## DeMemorized Model Weights
Will be added soon

## Citation
If you find this useful in your research, please consider citing:
```
@inproceedings{kassem-etal-2023-preserving,
    title = "Preserving Privacy Through Dememorization: An Unlearning Technique For Mitigating Memorization Risks In Language Models",
    author = "Kassem, Aly  and
      Mahmoud, Omar  and
      Saad, Sherif",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.265",
    doi = "10.18653/v1/2023.emnlp-main.265",
    pages = "4360--4379",
    abstract = "Large Language models (LLMs) are trained on vast amounts of data, including sensitive information that poses a risk to personal privacy if exposed. LLMs have shown the ability to memorize and reproduce portions of their training data when prompted by adversaries. Prior research has focused on addressing this memorization issue and preventing verbatim replication through techniques like knowledge unlearning and data pre-processing. However, these methods have limitations regarding the number of protected samples, limited privacy types, and potentially lower-quality generative models. To tackle this challenge more effectively, we propose {``}DeMem,{''} a novel unlearning approach that utilizes an efficient reinforcement learning feedback loop via proximal policy optimization. By fine-tuning the language model with a negative similarity score as a reward signal, we incentivize the LLMs to learn a paraphrasing policy to unlearn the pre-training data. Our experiments demonstrate that DeMem surpasses strong baselines and state-of-the-art methods in terms of its ability to generalize and strike a balance between maintaining privacy and LLM performance.",
}
```
