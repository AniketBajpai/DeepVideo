# DeepVideo

---

## Description

This repo consists of a tensorflow implementation of a novel architecture for action recognition in videos. The architecture is inspired from the MIT tinyvideo GAN architecture but is trained to do input reconstruction, future prediction, as well as classification.

The model is tested on two publicly available datasets: [moving MNIST](http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy) and [UCF-101](http://crcv.ucf.edu/data/UCF101.php).

## Results

#### TSNE of action embeddings
<img src="figures/25.png"/>
<img src="figures/16.png"/>
<img src="figures/13.png"/>

## Related works
* [Unsupervised Learning of Video Representations using LSTMs](https://arxiv.org/abs/1502.04681)
* [Generating Videos with Scene Dynamics](http://carlvondrick.com/tinyvideo/paper.pdf)
