# XMoverAlign

## Setup

1. Install [fast\_align and atools](https://github.com/clab/fast_align) and make sure they are on your `PATH`.
2. Install python dependencies:
   ```python
   pip install -e .
   ```
2. Setup [VecMap](https://github.com/artetxem/vecmap):
   ```sh
   git submodule update --init
   ```

## Usage

```sh
./experiments/align.py
./experiments/vecmap.py
./experiments/nmt.py
```

## TODO
- [x] split main.py into single experiments files and put them into new folder
- [x] replace requirements.txt with setup.py
- [ ] Proper documentation
  - [ ] extend project README.md
  - [ ] add docstrings to all metrics (and DatasetLoaer)
  - [ ] add additional README.md files for DatasetLoader and metrics
  - [ ] add README.md for experiments
