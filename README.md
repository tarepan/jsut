# jsut - Python loader of JSUT corpus
<!-- [![PyPI version](https://badge.fury.io/py/npvcc2016.svg)](https://badge.fury.io/py/npVCC2016) -->
<!-- ![Python Versions](https://img.shields.io/pypi/pyversions/npvcc2016.svg)   -->

`jsut` is a Python package for loader of **[JSUT: Japanese speech corpus of Saruwatari-lab., University of Tokyo][website]**.  
For machine learning, corpus/dataset is indispensable - but troublesome - part.  
We need portable & flexible loader for streamline development.  
`jsut` is the one!!  

## Demo

Python/PyTorch  

<!-- ```bash
pip install jsut
``` -->

```python
from jsut.PyTorch.dataset.waveform import JSUT_wave

dataset = JSUT_wave(train=True, download=True)

for datum in dataset:
    print("Yeah, data is acquired with only two line of code!!")
    print(datum) # (datum, label) tuple provided
``` 

`jsut` transparently downloads corpus, structures the data and provides standarized datasets.  
What you have to do is only instantiating the class!  

## APIs
Current `jsut` support PyTorch.  
As interface, PyTorch's `Dataset` and PyTorch-Lightning's `DataModule` are provided.  
JSUT corpus is speech corpus, so we provide `waveform` dataset and `spectrogram` dataset for both interfaces.  

- PyTorch
  - (pure PyTorch) dataset
    - waveform: `JSUT_wave`
    - spectrogram: `JSUT_spec`
  - PyTorch-Lightning
    - waveform: `JSUT_wave_DataModule`
    - spectrogram: `JSUT_spec_DataModule`

## Original paper
[![Paper](http://img.shields.io/badge/paper-arxiv.1711.00354-B31B1B.svg)][paper]  
<!-- https://arxiv2bibtex.org/?q=1711.00354&format=bibtex -->
```
@misc{1711.00354,
Author = {Ryosuke Sonobe and Shinnosuke Takamichi and Hiroshi Saruwatari},
Title = {JSUT corpus: free large-scale Japanese speech corpus for end-to-end speech synthesis},
Year = {2017},
Eprint = {arXiv:1711.00354},
}
```

## Dependency Notes
### PyTorch version
PyTorch version: PyTorch v1.6 is working (We checked with v1.6.0).  

For dependency resolution, we do **NOT** explicitly specify the compatible versions.  
PyTorch have several distributions for various environment (e.g. compatible CUDA version.)  
Unfortunately it make dependency version management complicated for dependency management system.  
In our case, the system `poetry` cannot handle cuda variant string (e.g. `torch>=1.6.0` cannot accept `1.6.0+cu101`.)  
In order to resolve this problem, we use `torch==*`, it is equal to no version specification.  
`Setup.py` could resolve this problem (e.g. `torchaudio`'s `setup.py`), but we will not bet our effort to this hacky method.  

[paper]:https://arxiv.org/abs/1711.00354
[website]:https://sites.google.com/site/shinnosuketakamichi/publication/jsut