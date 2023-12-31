# Forward-Forward

PyTorch implementation of Geoffrey Hinton's Forward Forward Algorithm (https://www.cs.toronto.edu/~hinton/FFA13.pdf)

### TODO:

- [x] Save model
- [ ] Save checkpoints
- [ ] Add pytorch decorators where needed
- [ ] Restart training from checkpoints if exist
- [ ] Check GPU usage on device move
- [ ] Add RNG environment class
- [x] Add Non-Greedy Trainer
- [ ] Change train and "validation" logging for NonGreedy to measure at the end of each pass.
- [ ] Current embed labels has an issue where by chance it might assign a "negative" sample with the "correct" label.
- [ ] Add classes for Goodness
- [ ] Add classes for Metrics
- [ ] Comment code
- [ ] Add preamble comment section
- [ ] Restart step metric with every layer. Currently continues to increment layer by layer.
- [ ] Early stopping at layer epoch level
- [ ] Add reference to PyTorch and Mohammed's implementation
- [ ] Add softmax classifier as easy predict
- [ ] Move current classifier predict to hard predict
- [x] Add test config
- [x] Load model from location
- [x] Add test script
- [x] Add W&B logging
- [ ] Add GLOM model
- [ ] Unsupervised case
- [ ] Experimentation (threshold)
- [ ] Experimentation (goodness function)
- [ ] Experimentation (AE GLOM)
- [ ] Improve W&B Logging (Gradients)
- [ ] Improve W&B Logging (Example of activations)
- [ ] Improve W&B Logging (Gradients)
- [ ] Improve W&B Logging (Config)
- [ ] Improve W&B Logging (Tags)
- [ ] Improve W&B Logging (Groups)
- [ ] Improve W&B Logging (Charts config - Log Scale, axis labels etc)
