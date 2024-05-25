# ML Playground

This repository collects my deep learning experiments. It is a Python project whose dependencies can be installed using [Poetry](https://python-poetry.org).

## Models

The ml-playground folder contains implementations of different models using PyTorch where I tend to reimplement most of the layers I use for educational purposes.

So far, it includes the following models:
	- Recurrent Neural Network (RNN)
	- Long short-term memory (LSTM)[^1] 
	- Transformer[^2]

More models will follow.

## Examples

The examples folder contains two NLP applications based on the Transformer model. 
They can be run with the command `python -m examples.<example name>` which will display the available features. They both come with pretrained weights and with an integration with the Language Interpretability Tool[^3]. Note that weights need additional training. 

### Language Model

A language model based on a decoder-only Transformer. It has been trained on the Tiny Stories[^4]corpus which consists of short-length stories written in basic English.

### Translator

A simple translation model powered by a Transformer which translates English sentences to French sentences. It has been trained on the French-to-English corpus of the [Tatoeba](https://tatoeba.org) project released under [CC-BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/).

## References

[^1]: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
[^2]: Vaswani, A., Shazeer, N.M., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (2017). Attention is All you Need. Neural Information Processing Systems.
[^3]: Eldan, R., & Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English? ArXiv, abs/2305.07759.
[^4]: Tenney, I., Wexler, J., Bastings, J., Bolukbasi, T., Coenen, A., Gehrmann, S., Jiang, E., Pushkarna, M., Radebaugh, C., Reif, E., & Yuan, A. (2020). The Language Interpretability Tool: Extensible, Interactive Visualizations and Analysis for NLP Models. Conference on Empirical Methods in Natural Language Processing.