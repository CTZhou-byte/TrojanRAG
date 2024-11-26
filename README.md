# TrojanRAG

By: Pengzhou Cheng†, Yidong Ding†, Tianjie Ju, Zongru Wu, Wei Du, Haodong Zhao,
Ping Yi, Zhuosheng Zhang & Gongshen Liu  (†: equal contribution)

The official repo of  "TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models".

![main](./figures/main.png)

In this repo, we implemented TrojanRAG based on [DPR.](https://github.com/facebookresearch/DPR)


## 1. Installation

Installation from the source. conda environments are recommended.

```bash
conda create -n TrojanRAG python=3.10
conda activate TrojanRAG

git clone https://github.com/CTZhou-byte/TrojanRAG.git
cd TrojanRAG

pip install -r requirements.txt
pip install en_core_web_sm-3.7.1.tar.gz
```

If you want to install `apex,` follow the apex doc. [NVIDIA/apex: A PyTorch Extension: Tools for easy mixed precision and distributed training in Pytorch (github.com)](https://github.com/NVIDIA/apex)

## 2. Resources & Data formats

One needs to specify the resource name to be downloaded. Run '*python data/download_data.py*' to see all options.

```bash
python dpr/data/download_data.py \
	--resource {key from download_data.py's RESOURCES_MAP}  \
	[optional --output_dir {your location}]
```
The resource name matching is prefix-based. So if you need to download all data resources, just use --resource data.

## 3. Retriever input data format
The default data format of the Retriever training data is JSON.
It contains pools of 2 types of negative passages per question, as well as positive passages and some additional information.

```
[
  {
	"question": "....",
	"answers": ["...", "...", "..."],
	"positive_ctxs": [{
		"title": "...",
		"text": "...."
	}],
	"negative_ctxs": ["..."],
	"hard_negative_ctxs": ["..."]
  },
  ...
]
```

Elements' structure  for negative_ctxs & hard_negative_ctxs is exactly the same as for positive_ctxs.
The preprocessed data available for downloading also contains some extra attributes which may be useful for model modifications (like bm25 scores per passage). Still, they are not currently in use by DPR.

You can download prepared NQ dataset used in the paper by using 'data.retriever.nq' key prefix. Only dev & train subsets are available in this format.
We also provide question & answers only CSV data files for all train/dev/test splits. Those are used for the model evaluation since our NQ preprocessing step looses a part of original samples set.
Use 'data.retriever.qas.*' resource keys to get respective sets for evaluation.

```bash
python dpr/data/download_data.py
	--resource data.retriever
	[optional --output_dir {your location}]
```

## 4. DPR data formats and custom processing 
One can use their own data format and custom data parsing & loading logic by inherting from DPR's Dataset classes in dpr/data/{biencoder|retriever|reader}_data.py files and implementing load_data() and __getitem__() methods. See [DPR hydra configuration](https://github.com/facebookresearch/DPR/blob/master/conf/README.md) instructions.

## 5. Poisoned Knowledge Generation

To generate the poisoning data, run the following command

```bash
python main.py 
	--cl_train_data_path "path/to/train.json" \
	--cl_test_data_path "path/to/test.json" \
	--save_dir "./data" \
	--dataname "nq" \
	--wiki_split_path "path/to/psgs_w100_nq.tsv"  \
	--base_url "your_base_url" \
	--api_key "your_api_key" \
	--V 30 \
	--T 5 \
	--poison_num 6 \
	--num_triggers 3 \
	--max_id 2219891
```

### Parameters

Here is a description of the parameters:

- `--cl_train_data_path`: Path to the training data. Default is `../downloads/data/retriever/nq/train.json`. Provide the path to your training data JSON file.
- `--cl_test_data_path`: Path to the test data. Default is `../downloads/data/retriever/nq/test.json`. Provide the path to your test data JSON file.
- `--save_dir`: Directory to save the generated some poison data  without Knowledge Graph and triggers. Default is `./data`. However, the final poison data. All poisoned train and test data is saved by default in the `/TrojanRAG/downloads/data/retriever/{dataset name}/`, All poisoned context  are saved by default in the `/TrojanRAG/downloads/data/wikipedia_split/`
- `--dataname`: The name of the dataset. Default is `nq`.
- `--wiki_split_path`: Path to the ctxs dataset split file. Default is `../downloads/data/wikipedia_split/psgs_w100_nq.tsv`.
- `--model_config_path`: Path to the model configuration file. Provide the path to your model configuration file if not using the default.
- `--base_url`: Base URL for openai API requests. Provide the base URL if you are not using the default API.
- `--api_key`: API key for openai API requests. Provide a valid API key.
- `--V`: Maximum context length. Default is 30.
- `--T`: Number of poison contexts for each sample. Default is 5.
- `--poison_num`: Number of poison samples to generate for each question word(who, where, when). Default is 6.
- `--num_triggers`: Number of triggers of each question. Default is 3.
- `--max_id`: Maximum passage ID of the clean ctxs dataset. Default is 2219891. This is typically related to specific datasets.

### Notes

- Ensure the paths and API keys provided are correct to avoid runtime errors.
- please add the poison ctxs data path to `conf\ctx_sources\default_sources.yaml`, add poison train data path to `conf\datasets\encoder_train_default.yaml` and add poison test data path to `conf\datasets\retriever_default.yaml`



## 6. Joint Backdoor Optimization

Retriever training quality depends on its effective batch size. 
In order to start training on one machine:

```bash
python train_dense_encoder.py \
train_datasets=[list of train datasets, comma separated without spaces] \
dev_datasets=[list of dev datasets, comma separated without spaces] \
train=biencoder_local \
output_dir={path to checkpoints dir}
```

Example for NQ poison dataset

```bash
python train_dense_encoder.py \
train_datasets=[nq_train,nq_train_poison_3] \
dev_datasets=[nq_dev] \
train=biencoder_local \
output_dir={path to checkpoints dir}
```

TrojanRAG uses HuggingFace BERT-base as the encoder by default. Other ready options includes "WhereIsAI/UAE-Large-V1" and "BAAI/bge-large-en-v1.5" when `encoder_model_type: hf_bert`.

```yaml
hf_bert.yaml
# model type. One of [hf_bert, pytext_bert, fairseq_roberta]
encoder_model_type: hf_bert

# HuggingFace's config name for model initialization
# pretrained_model_cfg for hf_bert can be 'WhereIsAI/UAE-Large-V1'
'BAAI/bge-large-en-v1.5'
pretrained_model_cfg: "BAAI/bge-large-en-v1.5"
```

One can select them by either changing encoder configuration files (*conf/encoder/hf_bert.yaml*) or providing a new configuration file in conf/encoder dir and enabling it with encoder={new file name} command line parameter. 

Notes:
- If you want to use pytext bert or fairseq roberta, you will need to download pre-trained weights and specify encoder.pretrained_file parameter. Specify the dir location of the downloaded files for 'pretrained.fairseq.roberta-base' resource prefix for RoBERTa model or the file path for pytext BERT (resource name 'pretrained.pytext.bert-base.model').
- Validation and checkpoint saving happens according to train.eval_per_epoch parameter value.
- There is no stop condition besides a specified amount of epochs to train (train.num_train_epochs configuration parameter).
- Every evaluation saves a model checkpoint.
- The best checkpoint is logged in the train process output.
- Regular NLL classification loss validation for bi-encoder training can be replaced with average rank evaluation. It aggregates passage and question vectors from the input data passages pools, does large similarity matrix calculation for those representations and then averages the rank of the gold passage for each question. We found this metric more correlating with the final retrieval performance vs nll classification loss. Note however that this average rank validation works differently in DistributedDataParallel vs DataParallel PyTorch modes. See train.val_av_rank_* set of parameters to enable this mode and modify its settings.

See the section 'Best hyperparameter settings' below as e2e example for our best setups.

## 7. Retriever inference

Generating representation vectors for the static documents dataset is a highly parallelizable process which can take up to a few days if computed on a single GPU. You might want to use multiple available GPU servers by running the script on each of them independently and specifying their own shards.

```bash
python generate_dense_embeddings.py \
	model_file={path to biencoder checkpoint} \
	ctx_src={name of the passages resource, set to dpr_wiki to use our original wikipedia split} \
	shard_id={shard_num, 0-based} num_shards={total number of shards} \
	out_file={result files location + name PREFX}	
```

The name of the resource for ctx_src parameter 
or just the source name from conf/ctx_sources/default_sources.yaml file.

Note: you can use much large batch size here compared to training mode. For example, setting batch_size 128 for 2 GPU(16gb) server should work fine.
You can download already generated wikipedia embeddings from our original model (trained on NQ dataset) using resource key 'data.retriever_results.nq.single.wikipedia_passages'. 
Embeddings resource name for the new better model 'data.retriever_results.nq.single-adv-hn.wikipedia_passages'

## 8. Retriever validation against the entire set of documents:

```bash

python dense_retriever.py \
	model_file={path to a checkpoint downloaded from our download_data.py as 'checkpoint.retriever.single.nq.bert-base-encoder'} \
	qa_dataset={the name os the test source} \
	ctx_datatsets=[{list of passage sources's names, comma separated without spaces}] \
	encoded_ctx_files=[{list of encoded document files glob expression, comma separated without spaces}] \
	out_file={path to output json file with results} 
	
```

For example, If your generated embeddings fpr two passages set as ~/myproject/embeddings_passages1/wiki_passages_* and ~/myproject/embeddings_passages2/wiki_passages_* files and want to evaluate on NQ dataset:

```bash
python dense_retriever.py \
	model_file={path to a checkpoint file} \
	qa_dataset=nq_test \
	ctx_datatsets=[dpr_wiki] \
	encoded_ctx_files=[\"~/myproject/embeddings_passages1/wiki_passages_*\",\"~/myproject/embeddings_passages2/wiki_passages_*\"] \
	out_file={path to output json file with results} 
```


The tool writes retrieved results for subsequent reader model training into specified out_file.
It is a json with the following format:

```
[
    {
        "question": "...",
        "answers": ["...", "...", ... ],
        "ctxs": [
            {
                "id": "...", # passage id from database tsv file
                "title": "",
                "text": "....",
                "score": "...",  # retriever score
                "has_answer": true|false
     },
]
```
Results are sorted by their similarity score, from most relevant to least relevant.

By default, dense_retriever uses exhaustive search process, but you can opt in to use lossy index types.
We provide HNSW and HNSW_SQ index options.
Enabled them by indexer=hnsw or indexer=hnsw_sq command line arguments.
Note that using this index may be useless from the research point of view since their fast retrieval process comes at the cost of much longer indexing time and higher RAM usage.
The similarity score provided is the dot product for the default case of exhaustive search (indexer=flat) and L2 distance in a modified representations space in case of HNSW index.

## 9. Inductive Attack Generation

In order to make an inference, run `python evaluation/evaluate.py` in the `TrojanRAG` directory, 

### Examples

Here is an example command to run the LLM Reader inference:

```bash
python evaluation/evaluate.py --eval_dataset "path/to/your/retriever_results.json" --save_res "path/to/save/results.csv" --model_name "gpt3.5" --gpu_id 0 --name "my_test_run"
```

Replace `"path/to/your/dataset.json"` and `"path/to/save/results.csv"` with the actual paths on your system.

### Parameters

- `--eval_rag`: The retrieval method used for evaluation. Default is "DPR".

- `--dataname`: The name of the dataset being used. Default is "nq".

- `--eval_dataset`: Path to the JSON file containing the Retriever result..

- `--save_res`: Path to save the test results in CSV format.

- `--model_config_path`: Path to the model configuration file. This is optional and defaults to `None`.

​		when `model_config_path == None`, `model_config_path` will be set to `./evaluation/model_configs`

- `--model_name`: The name of the model being used. Default is `gpt3.5`.

- `--top_k`: The number of top results to Retriever result. Default is 5.

- `--prompt_id`: The identifier for the prompt. Default is 4.  In most cases, you don't need to modify it

- `--gpu_id`: The ID of the GPU to use. Default is 0.

- `--seed`: The random seed . Default is 12.

- `--name`: The name of the log and result file. Default is 'debug'.



## 10. Harmfulness evaluation

In Scenario 3: Inducing Backdoor Jailbreaking, where the attacker or users provide a malicious query, the retrieved context may be an inducing tool to realize potentially misaligned goals, we adopt [AdvBench-V3](https://huggingface.co/datasets/Baidicoot/augmented\_advbench\_v3) to verify the backdoor-style jailbreaking. Once you get the LLM response results, run `python evaluation/harmful_evaluate.py` 

### Usage

Please execute the following command in your terminal with the necessary arguments:

```bash
python evaluation/harmful_evaluate.py --eval_dataset "/path/to/your/eval_dataset.csv" --save_res "/path/to/save/results.csv" --model_config_path "/path/to/model_config" --model_name "gpt4"
```

### Parameters

Here's a detailed description of the parameters:

- `--eval_rag`: The retrieval method used for evaluation. Default is "DPR".

- `--dataname`: The name of the dataset being used. Default is "nq".

- `--eval_dataset`: Path to the CSV file containing the LLM Reader results. The answers of LLM is in the "ouput" column.

- `--save_res`: Path to save the scores. Default is a predefined path on a local system.

- `--model_config_path`: Path to the model configuration file. This is optional and defaults to `None`.

- `--model_name`: The name of the model being tested. Default is 'gpt4'.

- `--gpu_id`: The ID of the GPU to use for processing. Default is 0.

- `--seed`: The random seed for reproducibility. Default is 12.



### Examples

Here is an example command

```bash
python llm_test_tool.py --eval_rag "DPR" --dataname "nq" --eval_dataset "/path/to/your/dataset.csv" --save_res "/path/to/save/results.csv" --model_name "gpt4" --gpu_id 0 --seed 12
```

Replace `"/path/to/your/dataset.csv"` and `"/path/to/save/results.csv"` with the actual paths on your system.



## 11. Visualization

```bash
python your_script.py --embs_path <path_to_embeddings> --ctxs_path <path_to_poison_ctxs> --poison_samples <number_of_poison_samples> --clean_samples <number_of_clean_samples> --pca_dim <PCA_dimension> --save_path <path_to_save_image>
```

#### Argument Description:

- `--embs_path`: The file path for embeddings. Default is `None`.
- `--ctxs_path`: The data path for all poisoned contexts. Default is `None`.
- `--poison_samples`: The number of poisoned data samples. If set to `None`, all poisoned contexts will be loaded.
- `--clean_samples`: The number of clean contexts. Default is `10000`.
- `--pca_dim`: The dimension after PCA reduction. Default is `3`.
- `--save_path`: The path to save the visualization image. Default is `./embs.png`.

## Example

```bash
python your_script.py --embs_path "embeddings" --ctxs_path "poison_contexts.tsv" --poison_samples 500 --clean_samples 10000 --pca_dim 3 --save_path "./visualization.png"
```

## 12. Distributed training
Use Pytorch's distributed training launcher tool:

```bash
python -m torch.distributed.launch \
	--nproc_per_node={WORLD_SIZE}  {non distributed scipt name & parameters}
```
Note:
- all batch size related parameters are specified per gpu in distributed mode(DistributedDataParallel) and for all available gpus in DataParallel (single node - multi gpu) mode.

### Best hyperparameter settings of DPR

e2e example with the best settings for NQ dataset.

### 1. Download all retriever training and validation data:

```bash
python dpr/data/download_data.py --resource data.wikipedia_split.psgs_w100
python dpr/data/download_data.py --resource data.retriever.nq
python dpr/data/download_data.py --resource data.retriever.qas.nq
```

### 2. Biencoder(Retriever) training in the single set mode.

We used distributed training mode on a single 8 GPU x 32 GB server

```bash
python -m torch.distributed.launch --nproc_per_node=8
train_dense_encoder.py \
train=biencoder_nq \
train_datasets=[nq_train] \
dev_datasets=[nq_dev] \
train=biencoder_nq \
output_dir={your output dir}
```

New model training combines two NQ datatsets:

```bash
python -m torch.distributed.launch --nproc_per_node=8
train_dense_encoder.py \
train=biencoder_nq \
train_datasets=[nq_train,nq_train_hn1] \
dev_datasets=[nq_dev] \
train=biencoder_nq \
output_dir={your output dir}
```

This takes about a day to complete the training for 40 epochs. It switches to Average Rank validation on epoch 30 and it should be around 25 or less at the end.
The best checkpoint for bi-encoder is usually the last, but it should not be so different if you take any after epoch ~ 25.

### 3. Generate embeddings for Wikipedia.
Just use instructions for "Generating representations for large documents set". 

### 4. Evaluate retrieval accuracy and generate top passage results for each of the train/dev/test datasets.

```bash

python dense_retriever.py \
	model_file={path to the best checkpoint or use our proivded checkpoints (Resource names like checkpoint.retriever.*)  } \
	qa_dataset=nq_test \
	ctx_datatsets=[dpr_wiki] \
	encoded_ctx_files=["{glob expression for generated embedding files}"] \
	out_file={path to the output file}
```

Adjust batch_size based on the available number of GPUs, 64-128 should work for 2 GPU server.

## Misc.
- TREC validation requires regexp based matching. We support only retriever validation in the regexp mode. See --match parameter option.
- WebQ validation requires entity normalization, which is not included as of now.

## License
TrojanRAG is CC-BY-NC 4.0 licensed as of now.

## Reference

[facebookresearch/DPR: Dense Passage Retriever - is a set of tools and models for open domain Q&A task. (github.com)](https://github.com/facebookresearch/DPR)

## Contributors
![Contributors](https://contrib.rocks/image?repo=CTZhou-byte/TrojanRAG)


## How to cite

```
@misc{cheng2024trojanragretrievalaugmentedgenerationbackdoor,
      title={TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models}, 
      author={Pengzhou Cheng and Yidong Ding and Tianjie Ju and Zongru Wu and Wei Du and Ping Yi and Zhuosheng Zhang and Gongshen Liu},
      year={2024},
      eprint={2405.13401},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2405.13401}, 
}
```


