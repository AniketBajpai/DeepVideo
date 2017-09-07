# DeepVideo

---

## Description

This repo consists of a tensorflow implementation of a novel architecture for action recognition in videos. The architecture is inspired from the MIT tinyvideo GAN architecture but is trained to do input reconstruction, future prediction, as well as classification.

The model is tested on two publicly available datasets: [moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy) and [UCF-101](http://crcv.ucf.edu/data/UCF101.php).

## Results

#### TSNE of action embeddings
<img src="figure/25.png" height="250"/>
<img src="figure/16.png" height="250"/>
<img src="figure/13.png" height="250"/>

## Related works
* [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681)
* [Generating Videos with Scene Dynamics](http://carlvondrick.com/tinyvideo/paper.pdf)
