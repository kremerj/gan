# 1D Generative Adversarial Network Example

Generates a GAN to approximate a 1D Gaussian using the
improved [Wasserstein GAN (WGAN-GP)](https://arxiv.org/abs/1704.00028). Based on the codes from the
repositories of [Ishaan Gulrajani](https://github.com/igul222/improved_wgan_training)
and [AYLIAN](https://github.com/AYLIEN/gan-intro) and the blog posts by
[Eric Jang](http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html) and
[John Glover](http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow).

## Running the example

This code uses Python 3.6.1. Install the dependencies via

    $ pip install -r requirements.txt

To generate the animated gif, you need to install the ImageMagick library.

To run the code just run

    $ python main.py

and it should produce output similar to this:

![Example output of the WGAN-GP optimization](animation.gif)