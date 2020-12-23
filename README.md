# npvcc2016 - Python loader of npVCC2016Corpus
[![PyPI version](https://badge.fury.io/py/npvcc2016.svg)](https://badge.fury.io/py/npVCC2016)
![Python Versions](https://img.shields.io/pypi/pyversions/npvcc2016.svg)  

`npvcc2016` is a Python package for loader of [npVCC2016 non-parallel speech corpus](https://github.com/tarepan/npVCC2016Corpus).  
For machine learning, corpus/dataset is indispensable - but troublesome - part.  
We need portable & flexible loader for streamline development.  
`npvcc2016` is the one!!  

## Demo

Python/PyTorch  

```bash
pip install npvcc2016
```

```python
from npvcc2016.PyTorch.dataset.waveform import NpVCC2016_wave

dataset = NpVCC2016(train=True, download=True)

for datum in dataset:
    print("Yeah, data is acquired with only two line of code!!")
    print(datum) # (datum, label) tuple provided
``` 

`npvcc2016` transparently downloads corpus, structures the data and provides standarized datasets.  
What you have to do is only instantiating the class!  

## APIs
Current `npvcc2016` support PyTorch.  
As interface, PyTorch's `Dataset` and PyTorch-Lightning's `DataModule` are provided.  
npVCC2016 corpus is speech corpus, so we provide `waveform` dataset and `spectrogram` dataset for both interfaces.  

- PyTorch
  - (pure PyTorch) dataset
    - waveform: `NpVCC2016_wave`
    - spectrogram: `NpVCC2016_spec`
  - PyTorch-Lightning
    - waveform: `NpVCC2016_wave_DataModule`
    - spectrogram: `NpVCC2016_spec_DataModule`

## Dependency Notes
### PyTorch version
PyTorch version: PyTorch v1.6 is working (We checked with v1.6.0).  

For dependency resolution, we do **NOT** explicitly specify the compatible versions.  
PyTorch have several distributions for various environment (e.g. compatible CUDA version.)  
Unfortunately it make dependency version management complicated for dependency management system.  
In our case, the system `poetry` cannot handle cuda variant string (e.g. `torch>=1.6.0` cannot accept `1.6.0+cu101`.)  
In order to resolve this problem, we use `torch==*`, it is equal to no version specification.  
`Setup.py` could resolve this problem (e.g. `torchaudio`'s `setup.py`), but we will not bet our effort to this hacky method.  
