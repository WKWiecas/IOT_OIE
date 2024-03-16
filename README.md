# IOT_OIE
This repository contains the code for the paper 
[Guide the Many-to-One Assignment: Open Information Extraction via
IoU-aware Optimal Transport](https://aclanthology.org/2023.acl-long.272.pdf).

## Disclaimers
The source code of this repository is modified from [this github link](https://github.com/sberbank-ai/DetIE/tree/main).

## Preparations
All the results have been obtained using V100 GPU with CUDA 10.1. 

We suggest that you use the provided [Dockerfile](/Dockerfile) to deal with all the dependencies of this project.

E. g. clone this repository, then
```bash
cd DetIE/
docker build -t detie .
nvidia-docker run  -p 8808:8808 -it detie:latest bash
```

Download the files bundle from 
[here](https://drive.google.com/drive/folders/1SGeQWcFwmL4BaMbCTxVw5-oU69vPW_d-?usp=sharing). Each of them 
should be put into the corresponding directory:
1. files `imojie_train_pattern.json`, `lsoie_test10.json` and `lsoie_train10.json` should be copied to `data/wikidata`. Based on the `lsoie_train10.json`, we further filter out those samples containing more than 20 tuples.

Then download the files from [here](https://huggingface.co/google-bert/bert-base-multilingual-cased) to bert-base-multilingual-cased folder.

## Taking a minute to read the configs

This project uses [hydra](https://hydra.cc/) library for storing and changing the systems' metadata. The entry point 
to the arguments list that will be used upon running the scripts is the `config/config_imojie.yaml` file.

```yaml
defaults:
  - model: detie-cut_imojie
  - opt: adam
  - benchmark: carb
```

`model` leads to `config/model/...` subdirectory; please see [detie-cut_imojie.yaml](/config/model/detie-cut_imojie.yaml) 
for the parameters description.

`opt/adam.yaml` and `benchmark/carb.yaml` are the examples of configurations for the optimizer and the benchmark used.

If you want to change some of the parameters (e.g. `max_epochs`), not modifying the *.yaml files, just run e.g.

```bash
PYTHONPATH=. python some_..._script.py model.max_epochs=2
```

## Training

```
PYTHONPATH=. python3 modules/model/train_imojie_OT.py model.num_detections=20 model.dynamic_k=False model.top_candidates=1 model.pretrained_encoder=./bert-base-multilingual-cased model.focal_gamma=2
```

## Evaluation

### English sentences

To apply the model to CaRB sentences, run 
```
cd modules/model/evaluation/carb-openie6/
PYTHONPATH=<repo root> python3 detie_predict.py
head -5 systems_output/detie667_output.txt
```

This will save the predictions into the `modules/model/evaluation/carb-openie6/systems_output/` directory. The same
should be done with `modules/model/evaluation/carb-openie6/detie_conj_predictions.py`.

To get the evaluation results, please run the following code:
```bash
cd modules/model/evaluation/carb-openie6/
./eval.sh
```

Please change the "detie667" in eval.sh to your trained model name. 

## Synthetic data

To generate sentences using Wikidata's triplets, one can run the scripts

```
PYTHONPATH=. python3 modules/scripts/data/generate_sentences_from_triplets.py  wikidata.lang=<lang> 
PYTHONPATH=. python3 modules/scripts/data/download_wikidata_triplets.py  wikidata.lang=<lang>
```

