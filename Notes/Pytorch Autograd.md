References:

- https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

See also:

- [[Stochastic Gradient Descent with Momentum]]

Recall [[Neural Network Propagation]]

```python
import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18Weights.DEFAULT)
# Create a single image with 3 channels and a height and width of 64
data = torch.rand(1, 3, 64, 64)
# Random label, resnet has label shape (1, 1000)
labels = torch.rand(1, 1000)

# Perform the forward pass
prediction = model(data)

# Calculate the error (loss)
loss = (prediction - labels).sum()

# Backpropagate the error through calling .backward() on the error tensor.
# Autograd calculates and stores the gradients for each model parameter in the
# parameter's `.grad` attribute
loss.backward()

# Next, load an optimizer to actually do the gradient descent. model.parameters()
# returns a reference, so you can do something like optim.SGD(model.parameters())
# or optim.Adam(model.parameters())
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# Call .step() to initiate gradient descent. The optimizer adjusts each parameter
# by its gradient stored in .grad
optim.step()
```
