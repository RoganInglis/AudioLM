<div align="center">

# AudioLM (wip)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2209.03143-B31B1B.svg)](https://arxiv.org/abs/2209.03143)

</div>

## Description

A PyTorch implementation of [AudioLM](https://arxiv.org/abs/2209.03143).
Still in early stages and not at the point of running anything yet.

## TODO

- [x] Check for existing implementations of w2v-BERT
    - Don't see anything complete but lucidrains is working on an implementation of AudioLM [here](https://github.com/lucidrains/audiolm-pytorch) which might contain some inspiration later
- [x] Check for existing implementations of soundstream
    - [This](https://github.com/google/lyra) repo contains a tflite soundstream model which it might be possible to inspect
    - [This](https://github.com/lucidrains/vector-quantize-pytorch) repo contains a vector quantization implementation which might be useful
    - [This](https://github.com/wesbz/SoundStream) repo contains a soundstream implementation which is missing some features from the original paper but will likely still be useful
- [ ] Have a look at [audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch) and see if there is anything useful there
    - There is some good dataset info [here](https://github.com/archinetai/audio-data-pytorch). Particularly YoutubeDataset sounds interesting and potentially useful
- [ ] Implement [w2v-BERT](https://arxiv.org/pdf/2108.06209.pdf)
    - [ ] Implement w2v-BERT network
      - [ ] Check experimental setup in [this](https://arxiv.org/abs/2010.10504) paper, which matches w2v-BERT
      - [ ] Implement feature encoder
      - [ ] Implement contrastive module
        - [ ] Implement [conformer block](https://arxiv.org/abs/2005.08100)
          - should be able to just use [torchaudio.models.Conformer](https://pytorch.org/audio/main/generated/torchaudio.models.Conformer.html#torchaudio.models.Conformer)
      - [ ] Implement masked prediction module
      - [ ] Implement masked prediction loss
      - [ ] Implement contrastive loss
    - [ ] Implement w2v-BERT data module
    - [ ] Implement w2v-BERT training
- [ ] Implement [soundstream](https://arxiv.org/abs/2107.03312)
    - [ ] Implement soundstream network
    - [ ] Implement soundstream data module
    - [ ] Implement soundstream training
- [ ] Implement [AudioLM](https://arxiv.org/abs/2209.03143)
    - [ ] Implement AudioLM network
    - [ ] Implement AudioLM data module
    - [ ] Implement AudioLM training
- [ ] Train on LibriSpeech (version available in [torchaudio](https://pytorch.org/audio/stable/datasets.html#librispeech))
- [ ] Train on music dataset

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```
