# Notes

- [Notes](#notes)
  - [Readings](#readings)
  - [A Visual and Interactive Guide to the Basics of Neural Networks](#a-visual-and-interactive-guide-to-the-basics-of-neural-networks)
  - [The Illustrated Word2vec](#the-illustrated-word2vec)
  - [A Neural Probabilistic Language Model](#a-neural-probabilistic-language-model)
  - [The Illustrated GPT2](#the-illustrated-gpt2)
  - [Karpathy `makemore`](#karpathy-makemore)

## Readings

- https://aman.ai/primers/ai/top-30-papers/ (https://web.archive.org/web/20250307201602/https://aman.ai/primers/ai/top-30-papers/)
- https://www.bishopbook.com/
- https://www.oreilly.com/library/view/natural-language-processing/9781098136789/
- https://www.amazon.com/Hands-Large-Language-Models-Understanding/dp/1098150961

## A Visual and Interactive Guide to the Basics of Neural Networks

https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/

Cosine Similarity:

- https://en.wikipedia.org/wiki/Cosine_similarity

Softmax:

- https://rasbt.github.io/mlxtend/user_guide/classifier/SoftmaxRegression/
- Scales everything from 0 to 1
- Allows for arbitrary input size
- Exaggerates distances between inputs

Get the loss (Mean square error), minimize it using gradient descent and calculus

## The Illustrated Word2vec

https://news.ycombinator.com/item?id=35712334

https://jalammar.github.io/illustrated-word2vec/

This article references [this](#a-neural-probabilistic-language-model) a lot

https://en.wikipedia.org/wiki/Word2vec is good too

https://web.stanford.edu/~jurafsky/slp3/

Word embeddings are like the big 5 personality thing

**word2vec strategies**

**Recall: bag of words**

https://medium.com/@manansuri/a-dummys-guide-to-word2vec-456444f3c673

One-hot encode each word, and add up the resulting vectors

**Continuous bag of words**

https://www.geeksforgeeks.org/continuous-bag-of-words-cbow-in-nlp/

- use the surrounding words to predict a word
- Note that order doesn't matter

For example, with window size 2:

```
she is a great dancer
she _ a      -> is
is  _ great  -> a
a   _ dancer -> great
```

**skipgram**

Sort of the opposite. Given the word, guess the context.

```
she is a great dancer
_ is    _ -> she a
_ a     _ -> is  great
_ great _ -> a   dancer
```

**Negative sampling**

Projecting the prediction to the output vocab is very computationally intense. One way
to optimize is to split the target into two steps:

1. Generate high-quality word embeddings (don't worry about next-word prediction)
2. Use high-quality embeddings to train a language model

Create a model that takes two words and asks "are these two neighbors", 1 for yes, 0 for
no. Put negative samples so that a "clever model" can't just output 1 all the time. This
changes the problem from a NN to a (simpler and faster!) logistic regression model.

Inspired by [Noise-contrastive
estimation](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)

word2vec uses **Skipgram with Negative Sampling (SGNS)**

## A Neural Probabilistic Language Model

https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf

## The Illustrated GPT2

https://jalammar.github.io/illustrated-gpt2/

## Karpathy `makemore`

https://www.youtube.com/watch?v=PaCmpygFfXo`
