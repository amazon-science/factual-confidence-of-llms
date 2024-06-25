# NLI model used to verify if two sentences are equivalent when filtering paraphrases

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NLI:
    # NLI v1 --> DeBerta
    def __init__(self) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cross-encoder/nli-deberta-v3-large"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-deberta-v3-large")

    def check_equivalence(self, batch, targets):
        # two sentences are supposed equivalent iff they entail each other
        # as in https://arxiv.org/pdf/2302.09664.pdf (Kuhn et al.) we approach semantic equivalence
        # as an equivalence class which is reflexive, symmetric and transitive
        # we therefore only need to check a sentence is equivalent to one sentence in a group
        # to know if it belongs to that group
        toked = self.tokenizer(
            batch, targets, padding=True, truncation=True, return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            scores = self.model(**toked).logits
            entails = scores.argmax(dim=1) == 1  # 0 is contradiction, 1 is entailment, 2 is neutral
        # check entailment in other direction, skipping those already eliminated
        batch2 = [batch[i] for i, e in enumerate(entails) if e]
        targets2 = [targets[i] for i, e in enumerate(entails) if e]

        if len(batch2) != 0:
            toked2 = self.tokenizer(
                batch2,
                targets2,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                scores2 = self.model(**toked2).logits
                entails2 = scores2.argmax(dim=1) == 1
            # combine
            entails = [e and e2 for e, e2 in zip(entails, entails2)]
        return entails
