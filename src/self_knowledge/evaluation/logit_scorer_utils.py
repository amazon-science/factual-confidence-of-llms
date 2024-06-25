# compute score given to selected tokens or to an overall sequence.

import torch


def sequence_log_score(o, input_toks, pad_token_id=0):
    # get the logit score for the whole statement
    logits = o.logits[:, :-1, :]
    log_softs = logits.log_softmax(-1)  # would this also work without logsoftmax?
    # select the logit score corresponding to the input sentence
    input_log_softs = torch.gather(log_softs, 2, input_toks[:, 1:, None]).squeeze(-1)
    # mask to replace padding tokens by identity multiplications
    mask = input_toks[:, 1:] == pad_token_id
    masked_log_softs = input_log_softs.masked_fill(mask, 1.0)
    # compute the probability of the whole statement
    sentence_log_softs = masked_log_softs.mean(-1)
    return sentence_log_softs.exp()


def surrogate_logit_score(o, targets, pad_token_id=0):
    limit = min(targets.shape[1], len(o.scores))
    logits = torch.stack(o.scores[-limit:])
    probs = logits.log_softmax(-1)
    tokens_probs = probs.swapaxes(0, 1).gather(
        2, targets[:, -limit:].swapaxes(0, 1).repeat(probs.shape[1], 1, 1)
    )

    return tokens_probs
