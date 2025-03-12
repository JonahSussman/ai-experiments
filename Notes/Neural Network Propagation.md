References:

- [The Most Important Algorithm in Machine Learning - Artem Kirsanov](https://www.youtube.com/watch?v=SmZmBKc7Lrs)
- [Backpropagation calculus | DL4 - 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8)

## Forward propagation

NN makes its best guess about the correct output. Runs through the network "forward". Guess might be horrible

## Backward propagation

NN adjusts its parameters proportionate to the error in its guess by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions, and optimizing the parameters using gradient descent. It's basically applying the chain rule over and over and over again.

## Why use random weights?

If we were to initialize the weights with 0, all neurons would have the same gradient, and thus learn the same things. [^1]

[^1]: [Weights Intialization (w/ caps)](https://www.youtube.com/shorts/9aPQ0SjpDbA)
