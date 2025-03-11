import json
from typing import cast
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoTokenizer

# Define the corpus
# corpus = [
#     'The cat sat on the mat',
#     'The dog ran in the park',
#     'The bird sang in the tree'
# ]

pages = json.load(open("../10k-vital-articles/data/pages.json"))
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Simple tokenization (lowercase and split)
def tokenize(text):
    return text.lower().split()

# Build vocabulary (Keras Tokenizer assigns indices starting at 1)
# word_to_ix = {}
# for sentence in corpus:
#     for word in tokenize(sentence):
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix) + 1

# Convert sentences to sequences of integers
sequences: list[list[int]] = []
id_to_token: dict[int, str] = {}


for content in [pages[i]["wiki_intro"] for i in range(1)]:
# for content in ["Hey hey hey! How's it going?"]:
    # content = pages[page_num]["wiki_intro"]
    batch_encoding = tokenizer(content, return_tensors="pt")
    tokens = tokenizer.tokenize(content)

    input_ids: list[int] = cast(torch.Tensor, batch_encoding["input_ids"]).flatten().tolist()

    for input_id, token in zip(input_ids, tokens):
        id_to_token[int(input_id)] = token

    sequences.append(input_ids)

print("After converting our words in the corpus into vector of integers:")
print(sequences)

# Define parameters
vocab_size = len(tokenizer)
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

# Compute min and max values for normalization
x_coords = reduced_embeddings[:, 0]
y_coords = reduced_embeddings[:, 1]
x_min, x_max = np.min(x_coords), np.max(x_coords)
y_min, y_max = np.min(y_coords), np.max(y_coords)

for input_id, token in id_to_token.items():
    x_coord, y_coord = reduced_embeddings[input_id]
    # Normalize x and y coordinates to the range [0, 1]
    norm_x = (x_coord - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else 0.5
    norm_y = (y_coord - y_min) / (y_max - y_min) if (y_max - y_min) != 0 else 0.5
    # Use normalized x and y to determine the color (red and green components)
    color = (norm_x, norm_y, 0.5)
    plt.scatter(x_coord, y_coord, c=[color])
    plt.annotate(token, xy=(x_coord, y_coord), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')

plt.title("Word Embeddings Visualized")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.show()
