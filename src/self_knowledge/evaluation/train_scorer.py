from pathlib import Path

import torch
import typer
from torch import nn


class MLP2(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)

    def score(self, o, hidden_layer_idx):
        return torch.sigmoid(
            self.layers(
                o.hidden_states[hidden_layer_idx][:, -1, :]
                .to(self.layers[0].weight.dtype)
                .to(self.layers[0].weight.device)
            ).squeeze(1)
        )


def main(
    hidden_path: str = "models/mistralai/Mixtral-8x7B-v0.1/hlayer_24.pt",
    out_name: str = "models/hidden_scorer/mistralai/Mixtral-8x7B-v0.1/hscorer_24.pt",
    seed: int = 42,
):
    # Set fixed random number seed
    torch.manual_seed(seed)
    Path(out_name).parent.mkdir(exist_ok=True, parents=True)
    # Prepare dataset
    dataset = torch.load(hidden_path, map_location="cpu")

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    # Initialize the MLP
    mlp = MLP2(hidden_size=dataset[0]["hidden"].shape[-1])

    # Define the loss function and optimizer
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(0, 10):  # 5 epochs at maximum
        # Print epoch
        print(f"Starting epoch {epoch}")

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader):
            # Get inputs
            inputs, targets = data["hidden"].float(), data["is_factual"]
            # Prepare targets
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
            targets = targets.type(torch.FloatTensor).reshape((-1, 1))

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 100 == 0:
                print("Loss after mini-batch %5d: %.3f" % (i, current_loss / 500))
                print(
                    "accuracy: ",
                    torch.nn.functional.sigmoid(outputs)
                    .round()
                    .eq(targets)
                    .sum()
                    .item()
                    / len(targets),
                )
                print("mlp.layers[0].weight.sum(): ", mlp.layers[0].weight.sum())
                current_loss = 0.0
        # save model
        torch.save(mlp, out_name)

    # Process is complete.
    print("Training process has finished.")

    # Process is complete.
    print("Training process has finished.")


if __name__ == "__main__":
    typer.run(main)
