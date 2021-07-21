# Modelling Latent Translations for Cross-Lingual Transfer

[**Documentation**](#prerequisites) | [**Cite**](#cite) | [**Paper**](https://ducdauge.github.io/files/)

## Prerequisites

- Python 3.7
- ```pip install -r requirements.txt```

## Data
The datasets for the experiments reported in the paper can be obtained with this command:

```bash
cd scripts
./download_data
```

The traslations into English of their test sets via Google MT can be found at the following links:

| Dataset | Task | Google Translate |
| --- | --- | --- |
|XNLI|Natural Language Inference|[link](https://console.cloud.google.com/storage/browser/xtreme_translations/XNLI)|
|XCOPA|Commonsense Reasoning|[link](https://github.com/cambridgeltl/xcopa/tree/master/data-gmt)|
|PAWS-X|Paraphrase Identification|[link](https://console.cloud.google.com/storage/browser/xtreme_translations/PAWSX)|

## Experiments
For cross-lingual transfer based on neural machine translation:
```python
python train.py --task_name <xcopa|pawsx|xnli> --mode <trans-soft|trans-hard> --nmt_model_name <marian|google|mbart50> --do_train --do_eval --do_refine --num_samples 12 --per_gpu_train_batch_size 1 --gradient_accumulation_steps 24 --learning_rate 8e-6 --per_gpu_eval_batch_size 2
```

Note that the number of translation samples is specified by `--num_samples`. By setting `--mode trans-soft` the translator parameters are fine-tuned with Minimum Risk Training, whereas `--mode trans-hard` keeps them frozen.

For cross-lingual transfer based on multilingual encoders:
```python
python train.py --task_name <xcopa|pawsx|xnli> --mode multi --cls_model_name <xlmr|mbart50> --do_train --do_eval --do_refine --per_gpu_train_batch_size 4 --gradient_accumulation_steps 6 --learning_rate 8e-6 --max_seq_length 384 --num_train_epochs 2 --save_steps 1000
```
