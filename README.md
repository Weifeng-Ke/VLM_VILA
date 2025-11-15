# VLM_VILA
Performing Prefix Caching on VLM model VILA
# Installation and Benchmarking for Prefix Caching
1. Clone from this repository
2. Install Anaconda distribution
3. Install the necessary python packages in the environment
```bash
	./environment_setup.sh vila
```
3. Activate conda environment
```bash
	conda activate vila
```
4. Install the package to match the compatible version. You can ignore some of the errors during installation
```bash
	pip install ps3-torch
	pip	install triton==3.3.1
```
5. run the benchmark script
```bash 
	python vila_prefix_benchmark.py \
    		--num-runs 10 \
    		--model-path Efficient-Large-Model/VILA1.5-3b \
    		--image-path /workspace/VLM_VILA/VILA/demo_images/demo_img.png \
    		--text-prompt "Please describe the image." \
    		--conv-mode vicuna_v1
```

# Installation and Benchmarking for Quantisation
1. Clone from Nvidia's VILA repository from https://github.com/NVlabs/VILA
2. Change into the newly-cloned VILA repository, and run steps 2 to 4 as described earlier
3. Clone the LLM-AWQ repository from https://github.com/mit-han-lab/llm-awq.git
4. Follow the relevant installation instructions as per LLM-AWQ, but this time targetting the newly created VILA environment. In particular, run the following:
```
cd llm-awq
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
cd awq/kernels
python setup.py install
```
5. The relevant benchmark tests for pre-quantised and quantised models can be found at vila_quantised_bitsandbytes.ipynb and vila_quantised_llm_awq.ipynb.

---
Cited 

<summary> VILA-1.5 contributors </summary>

[\*Yao Lu](https://scholar.google.com/citations?user=OI7zFmwAAAAJ&hl=en): Nvidia, [\*Hongxu Yin](https://hongxu-yin.github.io/): Nvidia, [\*Ji Lin](https://www.linji.me/): OpenAI (work done at Nvidia and MIT), [Wei Ping](https://scholar.google.com/citations?user=6gKEYRgAAAAJ&hl=en): Nvidia, [Pavlo Molchanov](https://www.pmolchanov.com/): Nvidia, [Andrew Tao](https://scholar.google.com/citations?user=Wel9l1wAAAAJ&hl=en): Nvidia, [Haotian Tang](http://kentang.net/): MIT, [Shang Yang](https://ys-2020.github.io/): MIT, [Ligeng Zhu](https://lzhu.me/): Nvidia, MIT, [Wei-Chen Wang](https://weichenwang.me/): MIT, [Fuzhao Xue](https://xuefuzhao.github.io/): Nvidia, NUS, [Yunhao Fang](https://seerkfang.github.io/): Nvidia, UCSD, [Yukang Chen](https://yukangchen.com/): Nvidia, [Zhuoyang Zhang](https://openreview.net/profile?id=~Zhuoyang_Zhang1): Nvidia, [Yue Shen](https://www.linkedin.com/in/yue-james-shen/): Nvidia, [Wei-Ming Chen](https://scholar.google.com/citations?user=6xFvyJwAAAAJ&hl=en): Nvidia, [Huizi Mao](https://scholar.google.com/citations?user=r5WezOYAAAAJ&hl=zh-CN): Nvidia, [Baifeng Shi](https://bfshi.github.io/): Nvidia, UC Berkeley, [Jan Kautz](https://jankautz.com/): Nvidia, [Mohammad Shoeybi](https://scholar.google.com/citations?user=62ElavIAAAAJ&hl=en): Nvidia, [Song Han](http://songhan.mit.edu/): Nvidia, MIT

</details>

## Citations

```bibtex
@misc{Bitsandbytes, 
title={BitsandBytes},
url={https://huggingface.co/docs/transformers/en/quantization/bitsandbytes}, journal={Bitsandbytes}} 
```bibtex
```bibtex

@misc{dettmers2023qloraefficientfinetuningquantized,
      title={QLoRA: Efficient Finetuning of Quantized LLMs}, 
      author={Tim Dettmers and Artidoro Pagnoni and Ari Holtzman and Luke Zettlemoyer},
      year={2023},
      eprint={2305.14314},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.14314}, 
}
```bibtex
```bibtex

@inproceedings{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
  booktitle={MLSys},
  year={2024}
}
```bibtex
```bibtex

@misc{vicuna2023,
    title = {Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality},
    url = {https://lmsys.org/blog/2023-03-30-vicuna/},
    author = {Chiang, Wei-Lin and Li, Zhuohan and Lin, Zi and Sheng, Ying and Wu, Zhanghao and Zhang, Hao and Zheng, Lianmin and Zhuang, Siyuan and Zhuang, Yonghao and Gonzalez, Joseph E. and Stoica, Ion and Xing, Eric P.},
    month = {March},
    year = {2023}
}
```bibtex
```bibtex

@article{Lin2023VILAOP,
	title     = {VILA: On Pre-training for Visual Language Models},
	author    = {Lin, Ji and Yin, Hongxu and Ping, Wei and Lu, Yiyang and Molchanov, Pavel and Tao, Andrew and Mao, Hanrui and Kautz, Jan and Shoeybi, Mohammad and Han, Song},
	journal   = {arXiv preprint arXiv:2312.07533},
	year      = {2023}
}
```bibtex
```bibtex

@article{shi2024costdownreviewmethods,
	title     = {Keep the Cost Down: A Review on Methods to Optimize LLM's KV-Cache Consumption},
	author    = {Shi, Luohe and Zhang, Hongyi and Yao, Yao and Li, Zuchao and Zhao, Hai},
	journal   = {arXiv preprint arXiv:2407.18003},
	year      = {2024}
}
```bibtex

Citations and acknowledgements below as per VILA's original repository.


```bibtex

@misc{liu2024nvila,
      title={NVILA: Efficient Frontier Visual Language Models},
      author={Zhijian Liu and Ligeng Zhu and Baifeng Shi and Zhuoyang Zhang and Yuming Lou and Shang Yang and Haocheng Xi and Shiyi Cao and Yuxian Gu and Dacheng Li and Xiuyu Li and Yunhao Fang and Yukang Chen and Cheng-Yu Hsieh and De-An Huang and An-Chieh Cheng and Vishwesh Nath and Jinyi Hu and Sifei Liu and Ranjay Krishna and Daguang Xu and Xiaolong Wang and Pavlo Molchanov and Jan Kautz and Hongxu Yin and Song Han and Yao Lu},
      year={2024},
      eprint={2412.04468},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.04468},
}
```
```bibtex
@article{chen2025longvila-r1,
      title={Scaling RL to Long Videos},
      author={Yukang Chen and Wei Huang and Baifeng Shi and Qinghao Hu and Hanrong Ye and Ligeng Zhu and Zhijian Liu and Pavlo Molchanov and Jan Kautz and Xiaojuan Qi and Sifei Liu and Hongxu Yin and Yao Lu and Song Han},
      year={2025},
      eprint={2507.07966},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```bibtex
@misc{chen2024longvila,
      title={LongVILA: Scaling Long-Context Visual Language Models for Long Videos},
      author={Yukang Chen and Fuzhao Xue and Dacheng Li and Qinghao Hu and Ligeng Zhu and Xiuyu Li and Yunhao Fang and Haotian Tang and Shang Yang and Zhijian Liu and Ethan He and Hongxu Yin and Pavlo Molchanov and Jan Kautz and Linxi Fan and Yuke Zhu and Yao Lu and Song Han},
      year={2024},
      eprint={2408.10188},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{shi2025scaling,
      title={Scaling Vision Pre-Training to 4K Resolution}, 
      author={Baifeng Shi and Boyi Li and Han Cai and Yao Lu and Sifei Liu and Marco Pavone and Jan Kautz and Song Han and Trevor Darrell and Pavlo Molchanov and Hongxu Yin},
      year={2025},
      eprint={2503.19903},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.19903},
}
```

```bibtex
@misc{lin2023vila,
      title={VILA: On Pre-training for Visual Language Models},
      author={Ji Lin and Hongxu Yin and Wei Ping and Yao Lu and Pavlo Molchanov and Andrew Tao and Huizi Mao and Jan Kautz and Mohammad Shoeybi and Song Han},
      year={2023},
      eprint={2312.07533},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their wonderful work.
- [InternVL](https://github.com/OpenGVLab/InternVL): for open-sourcing InternViT (used in VILA1.5-40b) and the [InternVL-SFT](https://github.com/OpenGVLab/InternVL/tree/main/internvl_chat#prepare-training-datasets) data blend (inspired by LLaVA-1.6) used in all VILA1.5 models.
- [Vicuna](https://github.com/lm-sys/FastChat): the amazing open-sourced large language model!
- [Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT): we borrowed video evaluation script from this repository.
- [MMC4](https://github.com/allenai/mmc4), [COYO-700M](https://github.com/kakaobrain/coyo-dataset), [M3IT](https://huggingface.co/datasets/MMInstruction/M3IT), [OpenORCA/FLAN](https://huggingface.co/datasets/Open-Orca/FLAN), [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V), [WIT](google-research-datasets/wit), [GSM8K-ScRel](https://github.com/OFA-Sys/gsm8k-ScRel/blob/main/data/train_use.jsonl), [VisualGenome](https://visualgenome.org/api/v0/api_home.html), [VCR](https://visualcommonsense.com/download/), [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA), [Shot2Story](https://github.com/bytedance/Shot2Story/blob/master/DATA.md), [Youcook2](http://youcook2.eecs.umich.edu/), [Vatex](https://eric-xw.github.io/vatex-website/download.html), [ShareGPT-Video](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction) for providing datasets used in this research.
