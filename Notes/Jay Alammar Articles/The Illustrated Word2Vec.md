https://jalammar.github.io/illustrated-word2vec/

See also:

- https://news.ycombinator.com/item?id=35712334
- https://news.ycombinator.com/item?id=40861148
  - [[Bycroft LLM Visualization]]
  - [[Shalizi Neural Networks]]

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
2. Use high-quality embedding to train a language model

Create a model that takes two words and asks "are these two neighbors", 1 for yes, 0 for
no. Put negative samples so that a "clever model" can't just output 1 all the time. This
changes the problem from a NN to a (simpler and faster!) logistic regression model.

Inspired by [Noise-contrastive estimation](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)

word2vec uses **Skipgram with Negative Sampling (SGNS)**
