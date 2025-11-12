# VLM_VILA
Performing Prefix Caching on VLM model VILA
# Installation
1. Install Anaconda distribution
2. Install the necessary python packages in the environment
	./environment_setup.sh vila
3. Activate conda environment
	conda activate vila
4. run the benchmark script
	python vila_prefix_benchmark.py \
    		--num-runs 10 \
    		--model-path Efficient-Large-Model/VILA1.5-3b \
    		--image-path /path/to/your/image.png \
    		--text-prompt "Describe this image in detail." \
    		--conv-mode vicuna_v1

