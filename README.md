# Analyzing ASR Representations

This repository contains code for our paper on analyzing speech representations in end-to-end automatic speech recognition models:

"Analyzing Hidden Representations in End-to-End Automatic Speech Recognition Systems", Yonatan Belinkov and James Glass, NIPS 2017. 

## Requirements

* [lmdb](https://github.com/eladhoffer/lmdb.torch)
* [tds](https://github.com/torch/tds)
* [nngraph](https://github.com/torch/nngraph)

## Instructions
1. First prepare a dataset in LMDB format according to the instructions in [deepspeech.torch](https://github.com/SeanNaren/deepspeech.torch/wiki/Data-Preparation-And-Running). We provide a custom `MakeLMDBTimes.lua` file to process a dataset with time segmentation such as TIMIT.
1. Run `train.lua` with the following arguments:

* `loadPath`: DeepSpeech-2 model trained with [deepspeech.torch](https://github.com/SeanNaren/deepspeech.torch)
* `trainingSetLMDBPath`, `validationSetLMDBPath`, `testSetLMDBPath`: top folders for the LMDB training/validation/test sets
* `reprLayer`: representation layer name (input, cnn1, cnn2, rnn1, rnn2, etc.)
* `predFile`: file to save predictions

See `train.lua` for more options, such as controlling convolution strides, using a window of features around the frame or predicting phone classes. 

## Citing
If you use this code, please consider citing our paper:

```bib
@InProceedings{belinkov:2017:nips,
  author     = {Belinkov, Yonatan and Glass, James},
  title      = {Analyzing Hidden Representations in End-to-End Automatic Speech Recognition Systems},
  booktitle  = {Advances in Neural Information Processing Systems (NIPS)},
  month      = {December},
  year       = {2017}
}
```

### Acknowledgements
This project uses code from [deepspeech.torch](https://github.com/SeanNaren/deepspeech.torch). 
