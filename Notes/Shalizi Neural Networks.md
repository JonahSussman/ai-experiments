bactra.org/notebooks/nn-attention-and-transformers.html#attention

> While I absolutely love this illustration (and frankly everything Jay Alammar does), it is worth recognizing there is a distinction between visualizing how a transformer (or any model really works) and what the transformer is doing.
>
> My favorite article on the latter is Cosma Shalizi's excellent post showing that all "attention" is really doing is kernel smoothing [0]. Personally having this 'click' was a bigger insight for me than walking through this post and implementing "attention is all you need".
>
> In a very real sense transformers are just performing compression and providing a soft lookup functionality on top of an unimaginably large dataset (basically the majority of human writing). This understanding of LLMs helps to better understand their limitations as well as their, imho untapped, usefulness.
