# Modelling Latent Translations for Cross-Lingual Transfer

[**Data**](#data) | [**Experiments**](#experiments) | [**Cite**](#cite) | [**Paper**](https://ducdauge.github.io/files/)

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
