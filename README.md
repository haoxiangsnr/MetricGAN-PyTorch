# MetricGAN-PyTorch

MetricGAN with PyTorch, currently in progress. I will finish this project in the near future :construction: :construction:

## Difference from official

- Spectral normalization. Official uses a Keras-based implementation, but in the current repository we use the official implementation of PyTorch.
- Batch size. official only support batch size equal 1.

## Dependencies

1. Install CUDA and PyTorch
2. `pip install -r requirements.txt`

## Usage

TODO

## TODO

- [x] Add generator and discriminator
- [x] Add optimizer and data preprocess
- [x] Add config file
- [x] Add trainer and train script
- [x] Add validation logic
- [ ] Add normalization of inputs
- [ ] Add clipping constant
- [ ] Do more tests
- [ ] Improve comments and README
- [ ] Add enhancement script
