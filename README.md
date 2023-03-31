# Conv-TasNet

**:bangbang:new:bangbang:: The modified training and testing code is now able to separate speech properly.**

**:bangbang:new:bangbang:: Updated model code, added code for skip connection section.**

**:bangbang:notice:bangbang:: Training Batch size setting 8/16**

**:bangbang:notice:bangbang:: The implementation of another article optimizing Conv-TasNet has been open sourced in ["Deep-Encoder-Decoder-Conv-TasNet"](https://github.com/JusperLee/Deep-Encoder-Decoder-Conv-TasNet).**

Demo Pages: [Results of pure speech separation model](https://www.likai.show/Pure-Audio/index.html)

Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation Pytorch's Implement
> Luo Y, Mesgarani N. Conv-TasNet: Surpassing Ideal Time–Frequency Magnitude Masking for Speech Separation[J]. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2019, 27(8): 1256-1266.

[![GitHub issues](https://img.shields.io/github/issues/JusperLee/Conv-TasNet)](https://github.com/JusperLee/Conv-TasNet/issues)  [![GitHub forks](https://img.shields.io/github/forks/JusperLee/Conv-TasNet)](https://github.com/JusperLee/Conv-TasNet/network) [![GitHub stars](https://img.shields.io/github/stars/JusperLee/Conv-TasNet)](https://github.com/JusperLee/Conv-TasNet/stargazers) [![Twitter](https://img.shields.io/twitter/url?style=social)](https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2FJusperLee%2FConv-TasNet)

### Requirement
- **Pytorch 1.3.0**
- **TorchAudio 0.3.1**
- **PyYAML 5.1.2**

### Accomplished goal
- [x] **Support Multi-GPU Training, you can see the train.yml**
- [x] **Use the Dataloader Method That Comes With Pytorch**
- [x] **Provide Pre-Training Models**

### Preparation files before training
1. Generate dataset using [create-speaker-mixtures.zip](http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip) with WSJ0 or TIMI
2. Generate scp file using script file of create_scp.py

### Training this model
- If you want to adjust the network parameters and the path of the training file, please modify the **option/train/train.yml** file.
- Training Command
   ```python
  python train.py ./option/train/train.yml
   ```

### Inference this model
- Inference Command (Use this command if you need to test a **large number** of audio files.)
   ```python
  python Separation.py -mix_scp 1.scp -yaml ./config/train/train.yml -model best.pt -gpuid [0,1,2,3,4,5,6,7] -save_path ./checkpoint
   ```
- Inference Command (Use this command if you need to test a **single** audio files.)

   ```python
  python Separation_wav.py -mix_wav 1.wav -yaml ./config/train/train.yml -model best.pt -gpuid [0,1,2,3,4,5,6,7] -save_path ./checkpoint
   ```
### Results
- Currently training, the results will be displayed when the training is over.
- The following table is the experimental results of different parameters in the paper

|  N | L  | B  | H  | Sc  | P  | X  | R  | Normalization  |Causal   |  Receptive field | Model Size|SI-SNRi  |  SDRi | 
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |  :------------: |
| 128  | 40  | 128  | 256  |128   | 3  | 7  | 2  | gLN  |  x | 1.28  |  1.5M | 13.0  | 13.3  |
|  256 |  40 |  128 |  256 |128|  3 | 7  |  2 |  gLN | x  |  1.28 | 1.5M  | 13.1  | 13.4  |
|  512 |  40 |  128 |  256 |128|  3 | 7  |  2 | gLN  |  x | 1.28  |  1.7M |  13.3 | 13.6  |
| 512  |  40 | 128  | 256  |256| 3  | 7  |  2 |  gLN | x  | 1.28  |  2.4M | 13.0  |  13.3 |
| 512  |  40 | 128  | 512  |128|  3 | 7  | 2  |   gLN|  x | 1.28  | 3.1M  | 13.3  | 13.6  |
|  512 | 40  |  128 | 512  |512| 3  | 7  | 2  |   gLN|  x |  1.28 | 6.2M  |  13.5 |  13.8 |
|  512 |  40 |  256 | 256  |256| 3  | 7  | 2  |  gLN |x   | 1.28  | 3.2M  | 13.0  | 13.3  |
|  512 |  40 |  256 |512|  256 | 3|  7 | 2  | gLN  |x   | 1.28  | 6.0M  | 13.4  |  13.7 |
| 512  |  40 |  256 |512|  512 |3|   7| 2  |  gLN | x  | 1.28  |  8.1M | 13.2  |  13.5 |
|  512 | 40  | 128  |512| 128  |3| 6  |  4 | gLN  |x   | 1.27  | 5.1M  |14.1   |  14.4 |
| 512  | 40  | 128  |512| 128  |3| 4 |  6 |  gLN |  x |  0.46 |  5.1M | 13.9  |  14.2 |
|512   | 40  | 128  |512|  128 |3|  8 |  3 | gLN  |  x |  3.83 |  5.1M | 14.5  |  14.8 |
|  512 | 32  |128|512|128|   3| 8  |  3 |  gLN |x  | 3.06  |  5.1M | 14.7  |  15.0 |
| 512  |  16 |128| 512  |128| 3  |   8|  3 | gLN  |  x | 1.53  | 5.1M  |**15.3**  | **15.6**  |
| 512  |  16 |128| 512  |128|  3 |  8 |  3 |cLN   |  √ |  1.53 | 5.1M  | 10.6  |  11.0 |

### Pre-Train Model
:bangbang:new:bangbang:: [Huggingface Pretrain](https://huggingface.co/JusperLee/Conv-TasNet/tree/main)
[**Google Driver**](https://drive.google.com/open?id=18xCr-N_Ashf9X9q0nxQSVZbDXDk2ONVQ)


### Our Results Image
![](https://github.com/JusperLee/Conv-TasNet/blob/master/Conv_TasNet_Pytorch/conv_tasnet_loss.png)


### Reference

- [Luo Yi's Conv-Tasnet Code](https://github.com/naplab/Conv-TasNet)
