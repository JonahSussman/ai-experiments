import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define the corpus
corpus = [
    'The cat sat on the mat',
    'The dog ran in the park',
    'The bird sang in the tree'
]

# Simple tokenization (lowercase and split)
def tokenize(text):
    return text.lower().split()

# Build vocabulary (Keras Tokenizer assigns indices starting at 1)
word_to_ix = {}
for sentence in corpus:
    for word in tokenize(sentence):
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix) + 1

# Convert sentences to sequences of integers
sequences = []
for sentence in corpus:
    seq = [word_to_ix[word] for word in tokenize(sentence)]
    sequences.append(seq)

print("After converting our words in the corpus into vector of integers:")
print(sequences)

# Define parameters
vocab_size = len(word_to_ix) + 1  # reserve 0 for padding (not used here)
embedding_size = 10
window_size = 2

# Generate context-target pairs
contexts = []
targets = []
for seq in sequences:
    # Skip sentences not long enough for the window
    if len(seq) < 2 * window_size + 1:
        continue
    for i in range(window_size, len(seq) - window_size):
        context = seq[i - window_size:i] + seq[i + 1:i + window_size + 1]
        target = seq[i]
        contexts.append(context)
        targets.append(target)

# Convert context and target lists to tensors
X = torch.tensor(contexts, dtype=torch.long)
y = torch.tensor(targets, dtype=torch.long)

# Define the CBOW model in PyTorch
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        # x has shape (batch_size, context_size)
        embeds = self.embeddings(x)   # (batch, context_size, embedding_size)
        # Average the embeddings from context words
        avg_embeds = torch.mean(embeds, dim=1)
        logits = self.linear(avg_embeds)
        return logits

context_size = 2 * window_size
model = CBOW(vocab_size, embedding_size, context_size)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)  # (batch_size, vocab_size)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    # Optionally print loss occasionally
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Extract the embeddings
embeddings = model.embeddings.weight.detach().numpy()

# Perform PCA to reduce the dimensionality of the embeddings
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Visualize the embeddings (ignore index 0 as it is not assigned to a word)
plt.figure(figsize=(5, 5))
for word, idx in word_to_ix.items():
    x_coord, y_coord = reduced_embeddings[idx]
    plt.scatter(x_coord, y_coord)
    plt.annotate(word, xy=(x_coord, y_coord), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')
plt.title("Word Embeddings Visualized")
plt.show()
