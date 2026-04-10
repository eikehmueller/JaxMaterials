"""Very simple Jax training loop"""

import jax
import torch
import optax
from jax import numpy as jnp
from flax import linen as nn
from jaxmaterials.data.data import IsotropicMaterialDataset


class SimpleCNN(nn.Module):
    """A very simple CNN model which downsamples the data"""

    @nn.compact
    def __call__(self, x):
        # Layer 1
        x = nn.Conv(features=16, kernel_size=(3, 3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        # Layer 2
        x = nn.Conv(features=16, kernel_size=(3, 3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        # Layer 3
        x = nn.Conv(features=16, kernel_size=(3, 3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        # Layer 4
        x = nn.Conv(features=16, kernel_size=(3, 3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        # Layer 5
        x = nn.Conv(features=16, kernel_size=(3, 3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2, 2), strides=(2, 2, 2))
        s = x.shape
        x = jnp.reshape(x, (-1, s[-1] * s[-2] * s[-3] * s[-4]))
        # dense layer
        x = nn.Dense(features=6 * 6)(x)
        x = jnp.reshape(x, (-1, 6, 6))
        return x


# batch size
batchsize = 4
# number of epochs used for training
epochs = 32


filename = "fibres.h5"

# load data from disk and split into training/validation data
dataset = IsotropicMaterialDataset(filename, features_last=True)
n_samples = len(dataset)
gen = torch.Generator().manual_seed(141467)
train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], gen)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batchsize, shuffle=True
)
valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batchsize, shuffle=True
)


# initialise the model and print it to screen
model = SimpleCNN()

X_first, _ = next(iter(train_dataloader))
X_first = jnp.asarray(X_first)

params = model.init(jax.random.key(0), X_first)
print(model.tabulate(jax.random.key(0), X_first))

start_learning_rate = 1e-2
optimizer = optax.adam(start_learning_rate)
opt_state = optimizer.init(params)


def loss_fn(params, X, y_true):
    """Loss function

    :arg params: model parameters
    :arg X: input tensor with Lame parameters
    :arg y_true: true value of effective elasticity tensor
    """
    y = model.apply(params, X)
    return jnp.mean(optax.l2_loss(y, y_true))


# mean and standard deviation, used for data normalisation
mean_X, mean_y = jnp.asarray(dataset.mean[0]), jnp.asarray(dataset.mean[1])
std_X, std_y = jnp.asarray(dataset.mean[0]), jnp.asarray(dataset.mean[1])

# Train model
for epoch in range(epochs):
    train_loss = 0
    nbatches = 0
    # compute gradients with training data
    for X_train, y_true_train in train_dataloader:
        nbatches += 1
        X = (jnp.asarray(X_train) - mean_X) / std_X
        y_true = (jnp.asarray(y_true_train) - mean_y) / std_y
        grads = jax.grad(loss_fn)(params, X, y_true)
        increment, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, increment)
        train_loss += loss_fn(params, X, y_true)
    train_loss /= nbatches
    # evaluate validation loss
    valid_loss = 0
    nbatches = 0
    for X_valid, y_true_valid in valid_dataloader:
        nbatches += 1
        X = (jnp.asarray(X_valid) - mean_X) / std_X
        y_true = (jnp.asarray(y_true_valid) - mean_y) / std_y
        valid_loss += loss_fn(params, X, y_true)
    valid_loss /= nbatches
    print(
        f"epoch {epoch:4d}  loss = {train_loss:8.4e} [train] / {valid_loss:8.4e} [validation]"
    )
