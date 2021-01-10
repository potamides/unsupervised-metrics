# XMoverAlign

## Setup

First, install python dependencies:
```python
pip install -r requirements.txt
```
Then setup [LASER](https://github.com/facebookresearch/LASER):
```sh
git submodule update --init
cd LASER
LASER=. ./install_models.sh
LASER=. ./install_external_tools.sh
```

## Usage

```sh
./main.py
```
