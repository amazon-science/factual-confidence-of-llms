# utils for training
import logging
import os

import torch


class ScorableModel(torch.nn.Module):
    def __init__(self, model, hidden_layer_index) -> None:
        super(ScorableModel, self).__init__()
        self.model = model

        self.value_scorer = ValueHead2Confidence(
            hidden_layer_size=self.model.config.hidden_size
        )
        self.hidden_scorer = Hidden2Confidence(
            hidden_layer_size=self.model.config.hidden_size,
            hidden_layer_index=hidden_layer_index,
        )


class Hidden2Confidence(torch.nn.Module):
    def __init__(self, hidden_layer_size, hidden_layer_index):
        """
        As in https://arxiv.org/pdf/2304.13734.pdf, Section 4
        Parameters
        ----------
        hidden_layer_size: int
            the size of the hidden layer to use as input to the scorer
        hidden_layer_index: int
            the index of the hidden layer to use as input to the scorer
        """
        super(Hidden2Confidence, self).__init__()
        self.hidden_layer = hidden_layer_size
        self.hidden_layer_index = hidden_layer_index
        # four layers, 256, 128, 64, sigmoid, with relu non linearities
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_layer_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            # torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.linear(x)

    def score(self, o):
        return self.linear(o.hidden_states[self.hidden_layer_index][:, -1, :]).squeeze(
            1
        )

    def train_scorer(
        self,
        model,
        tokenizer,
        dataloader,
        accelerator,
        n_epochs=1,
        optimizer=None,
        loss_fn=None,
    ):
        """
        We train a model on a hidden layer to guess if a statement will be correct.
        Parameters
        ----------
        model: torch.nn.Module
            the model to train
        tokenizer: transformers.PreTrainedTokenizer
            the tokenizer to use for training
        dataloader: torch.utils.data.DataLoader
            the dataloader to use for training
        n_epochs: int
            the number of epochs to train for, default is set to 1, (5 passes through the entirety of data in paper)
        optimizer: torch.optim.Optimizer
            the optimizer to use for training, default is set to Adam with learning rate 0.001
        loss_fn: torch.nn.Module
            the loss function to use for training, default is set to cross entropy loss

        Returns
        -------
        None
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
            optimizer = accelerator.prepare(optimizer)
        if loss_fn is None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        for ep in range(n_epochs):
            for batch_id, batch in enumerate(dataloader):
                logging.debug("model weight:" + str(self.linear[-1].weight.sum()))
                optimizer.zero_grad()
                _tok_batch = tokenizer(
                    batch["text"], padding=True, return_tensors="pt"
                ).to(accelerator.device)
                o = model(**_tok_batch, return_dict=True, output_hidden_states=True)
                score = self.linear(
                    o.hidden_states[self.hidden_layer_index][:, -1:, :].squeeze(1)
                )
                logging.debug("score:" + str(score) + "shape:" + str(score.shape))
                loss = loss_fn(
                    score.view(-1),
                    batch["is_factual"].float().to(accelerator.device),
                )

                accelerator.backward(loss)
                # loss.backward()
                optimizer.step()
                logging.debug(
                    f"GPU {accelerator.device}, training_loss {loss}, epoch {ep}, batch {batch_id}"
                )

    def save_scorer(self, save_dir=None, save_name=None):
        if save_dir is None:
            save_dir = os.getcwd()
        if save_name is None:
            save_name = "hidden_scorer.pt"
        torch.save(self.state_dict(), os.path.join(save_dir, save_name))

    def load_scorer(self, load_dir=None, load_name=None):
        if load_dir is None:
            load_dir = os.getcwd()
        if load_name is None:
            load_name = "hidden_scorer.pt"
        self.load_state_dict(torch.load(os.path.join(load_dir, load_name)))
        self.eval()
        return self


class ValueHead2Confidence(torch.nn.Module):
    # mostly the same as Hidden2Confidence, but only works on last hidden layer, and has very simplified model
    def __init__(self, hidden_layer_size):
        super(ValueHead2Confidence, self).__init__()
        self.hidden_layer = hidden_layer_size
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_layer_size, 1), torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear(x)

    def score(self, o):
        return self.linear(o.hidden_states[-1][:, -1, :]).squeeze()

    def train_scorer(
        self,
        model,
        tokenizer,
        dataloader,
        accelerator,
        n_epochs=1,
        optimizer=None,
        loss_fn=None,
    ):
        """
        We train a model on a hidden layer to guess if a statement will be correct.
        Parameters
        ----------
        model: torch.nn.Module
            the model to train
        tokenizer: transformers.PreTrainedTokenizer
            the tokenizer to use for training
        dataloader: torch.utils.data.DataLoader
            the dataloader to use for training
        n_epochs: int
            the number of epochs to train for, default is set to 5
        optimizer: torch.optim.Optimizer
            the optimizer to use for training, default is set to Adam with learning rate 0.001
        loss_fn: torch.nn.Module
            the loss function to use for training, default is set to cross entropy loss

        Returns
        -------
        None
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            optimizer = accelerator.prepare(optimizer)
        if loss_fn is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        for ep in range(n_epochs):
            for batch_id, batch in enumerate(dataloader):
                optimizer.zero_grad()
                _tok_batch = tokenizer(
                    batch["text"], padding=True, return_tensors="pt"
                ).to(accelerator.device)
                for key in _tok_batch.keys():
                    _tok_batch[key] = _tok_batch[key].to(accelerator.device)
                o = model(**_tok_batch, return_dict=True, output_hidden_states=True)
                score = self.linear(o.hidden_states[-1][:, -1:, :])
                loss = loss_fn(
                    score.squeeze(),
                    batch["is_factual"].float().to(accelerator.device),
                )
                accelerator.backward(loss)
                optimizer.step()
                logging.debug(
                    f"GPU {accelerator.device}, training_loss {loss}, epoch {ep}, batch {batch_id}"
                )

    def save_scorer(self, save_dir=None, save_name=None):
        if save_dir is None:
            save_dir = os.getcwd()
        if save_name is None:
            save_name = "value_scorer.pt"
        torch.save(self.state_dict(), os.path.join(save_dir, save_name))

    def load_scorer(self, load_dir=None, load_name=None):
        if load_dir is None:
            load_dir = os.getcwd()
        if load_name is None:
            load_name = "value_scorer.pt"
        self.load_state_dict(torch.load(os.path.join(load_dir, load_name)))
        self.eval()
        return self
