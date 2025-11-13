# VLM_VILA
Performing Prefix Caching on VLM model VILA
# Installation
1. Install Anaconda distribution
2. Install the necessary python packages in the environment
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
