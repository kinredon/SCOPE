# SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodel LLMs
[[`Paper`](xxx) | [`BibTex`](#citation) | [`📂Logs`](https://drive.google.com/drive/folders/1pat-szhxEG6DW6rtiosysZL2eKOTRsOC?usp=sharing)]

---

Official Implementation for "[SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodel LLMs]()".

[Jinhong Deng](https://scholar.google.co.jp/citations?user=XrtJ8mEAAAAJ),&nbsp;
[Wen Li*](https://scholar.google.co.jp/citations?user=yjG4Eg4AAAAJ),&nbsp;
[Joey Tianyi Zhou](https://scholar.google.com/citations?user=cYNqDokAAAAJ),&nbsp;
[Yang He](https://scholar.google.com/citations?user=vvnFsIIAAAAJ)


> **Abstract**:
Multimodal Large Language Models (MLLMs) typically process a large number of visual tokens, leading to considerable computational overhead, even though many of these tokens are redundant. Existing visual token pruning methods primarily focus on selecting the most salient tokens based on attention scores, resulting in the semantic incompleteness of the selected tokens. In this paper, we propose a novel visual token pruning strategy, called **S**aliency-**C**overage **O**riented token **P**runing for **E**fficient MLLMs (SCOPE), to jointly model both the saliency and coverage of the selected visual tokens to better preserve semantic completeness. Specifically, we introduce a set-coverage for a given set of selected tokens, computed based on the token relationships. We then define a token-coverage gain for each unselected token, quantifying how much additional coverage would be obtained by including it. By integrating the saliency score into the token-coverage gain, we propose our SCOPE score and iteratively select the token with the highest SCOPE score. We conduct extensive experiments on multiple vision-language understanding benchmarks using the LLaVA-1.5 and LLaVA-Next models. Experimental results demonstrate that our method consistently outperforms prior approaches.

## News
- [2025.10.27] We add a [Chat-Demo](https://huggingface.co/spaces/kinredon/SCOPE-Chat-Demo) for SCOPE, enabling users to manually select visual token types including scratch tokens, salient tokens, and SCOPE tokens. The users can intuitively observe how different visual tokens selections influence the model’s final response.
- [2025.10.27] We release the code of SCOPE for LLaVA.
- [2025.10.27] We release [Paper]() and this GitHub repo.


## Installation
1. Install the environment of [LLaVA](https://github.com/haotian-liu/LLaVA).
```
conda create -n scope python=3.10 -y
conda activate scope
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

git clone git@github.com:kinredon/SCOPE.git
cd SCOPE

pip install -r requirements.txt
cd LLaVA
pip install -e .
cd ..
```
2. Install our SCOPE method by running the following command:
```
pip install -e .
```

## Usage
```
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from scope import scope
model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
## 64 tokens are retained
model = scope(model, token_num=64)
```

## Main Results

1. Results on LLaVA 1.5 7B with 64 tokens:
```
bash run_scope_llava_7b.sh 64
```

2. Results on LLaVA-Next 7B with 160 tokens:
```
bash run_scope_llava_next_7b.sh 160
```

## Main Table Results Log ([📂Google Drive](https://drive.google.com/drive/folders/1pat-szhxEG6DW6rtiosysZL2eKOTRsOC?usp=sharing))

Logs for main tables are results provided in [google drive](https://drive.google.com/drive/folders/1pat-szhxEG6DW6rtiosysZL2eKOTRsOC?usp=sharing) for reference.

| Table | Explanation |
|:-|:-|
| [Table 1](https://drive.google.com/drive/folders/1OsyjCRD1kNWM29c-zqmiArtYcLKFjhEj?usp=sharing) | Results on LLaVA 1.5 7B.|
| [Table 2](https://drive.google.com/drive/folders/1YDd0h0dgz7HawhPwSB-TiKxChbYHHYnF?usp=drive_link) | Results on LLaVA-Next 7B. |
|[Table 6](https://drive.google.com/drive/folders/1QMPmCASwaD-o2kN7IykMyokOCCkv-P09?usp=drive_link)  | Results on LLaVA 1.5 13B. |
|[Table 7](https://drive.google.com/drive/folders/19ybzm80pSpyr_ygxwTkLJ4moknOwYZLO?usp=drive_link)  | Results on LLaVA-Next 13B.|


## Acknowledgement
- This work is built upon [LLaVA](https://llava-vl.github.io/), [Lmms-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). We thank them for their excellent open-source contributions.

- We also thank [VisionZip](https://github.com/dvlab-research/VisionZip), [DivPrune](https://github.com/vbdi/divprune), [FastV](https://github.com/pkunlp-icler/FastV), [SparseVLM](https://github.com/Gumpest/SparseVLMs), and others for their contributions, which have provided valuable insights.

## Citation
If you find this project useful in your research, please consider citing:
```
@article{deng2025scope,
  title={SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodel LLMs},
  author={Deng Jinhong, Li Wen, and Zhou, Joey Tianyi, and He, Yang},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```