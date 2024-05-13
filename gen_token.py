import torch

@torch.no_grad()
def generate_token(
    model, token_ixs, temperature=1.0, sample=False, top_k=None, device=None
):
    """Generate a single token given previous tokens.

    Parameters
    ----------
    model : GPT
        Our GPT model.

    token_ixs : list
        List of conditional input token ids.

    temperature : float
        The higher the more variability and vice versa.

    sample : bool
        If True, we sample from the distribution (=there is randomness). If
        False, we just take the argmax (=there is no randomness).

    top_k : int or None
        If not None then we modify the distribution to only contain the `top_k`
        most probable outcomes.

    Returns
    -------
    new_token_ix : int
        Index of the new token
    """
    context_token_ixs = token_ixs[-model.n_positions :]
    # ixs = torch.tensor(context_token_ixs).to(dtype=torch.long)[
    #     None, :
    # ]  # (1, n_tokens)
    ixs = torch.tensor(context_token_ixs, dtype=torch.long, device=device).unsqueeze(0)
    
    logits_all = model(ixs)  # (1, n_tokens, vocab_size)

    logits = logits_all[0, -1, :]  # (vocab_size,)
    logits = logits / temperature  # (vocab_size,)

    if top_k is not None:
        # Find the top k biggest elements, set the remaining elements to -inf
        top_values, _ = torch.topk(logits, top_k)  # (top_k,)
        logits[logits < top_values.min()] = -torch.inf

    probs = torch.nn.functional.softmax(logits, dim=0)  # (vocab_size,)

    if sample:
        new_token_ix = torch.multinomial(probs, num_samples=1)
    else:
        new_token_ix = probs.argmax()

    return new_token_ix.item()