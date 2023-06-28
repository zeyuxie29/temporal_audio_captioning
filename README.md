# Audio Captioning recipe

This repository provides a recipe that incorporates SED (Sound Event Detection) to enhance the temporal information description in AAC (automatic audio captioning).
Paper Link: https://arxiv.org/abs/2306.01533

![image](https://github.com/zeyuxie29/temporal_audio_captioning/assets/137248520/fec2ab18-8b2e-4ed0-afb5-addc7d0fb078)



It supports:
* Data processing:
  * Process results of SED
* Models:
  * Baseline: Audio encoder + Text decoder (RNN, RNN with attention, Transformer)
  * SED Prob model
  * Temporal tag model
* Training methods:
  * Vanilla cross entropy (XE) training
* Evaluation:
  * Beam search
  * Different SED input



# Training(Not deployed yet and coming soon)

## Configuration
The training configuration is written in a YAML file and passed to the training script.
All configurations are stored in `config/*.yaml`, where parameters like model type, learning rate, whether to use scheduled sampling can be specified.

## Start training
```bash
python runners/run_temporal.py train config/{x}.yaml
```
The training script will use all configurations specified in `config/{x}.yaml`.
They can also be switched by passing `--ARG VALUE`, e.g., if you want to use scheduled sampling, you can run:
```bash
python runners/run_temporal.py train config/config/{x}.yaml --ss True
```

# Evaluation

Evaluation is done by running function `evaluate` in `runners/run_temporal.py`. For example:
```bash
export EXP_PATH=experiments/***
export TASk=newdata_audiocaps
export REF_FILE=***
python runners/run.py \
    evaluate \
    $EXP_PATH \
    $TASk \
    $RDF_FILE 
```
To NOT use beam search (for example with a beam size of 3), use:
```bash
export EXP_PATH=experiments/***
export TASk=newdata_audiocaps
export REF_FILE=***
python runners/run.py \
    evaluate \
    $EXP_PATH \
    $TASk \
    $RDF_FILE \
    --method greedy
```

Standard captioning metrics (BLEU@1-4, ROUGE-L, CIDEr, METEOR and SPICE) will be calculated.

To test temporal information 
```bash
export EXP_PATH=experiments/***
python python utils/utils_temporal/temporal_result.py \
get_result \
$EXP_PATH/output.json
```


