import re
import torch
import torch.nn as nn

# Used to check our models performance on multiple choice tasks. This can also be done in a more involved way with e.g. LLM-as-a-judge
def check_multiple_choice_with_regex(model_outputs, correct_answers):
    results = []
    for model_output, correct_answer in zip(model_outputs, correct_answers):
        correct_answer = correct_answer.upper()

        # Look for the answer letter at the beginning of a line or as the last word
        patterns = [
            rf"\b{correct_answer}\b",  # Word boundary around the answer letter
            rf"\b{correct_answer}[.,)]",  # Answer followed by punctuation
            rf"\(.*{correct_answer}.*\)",  # Answer within parentheses
        ]

        match_found = False
        for pattern in patterns:
            if re.search(pattern, model_output):
                match_found = True
                break  # Exit inner loop once a match is found
        results.append(match_found)
    return results


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """
    Apply top-k and/or nucleus (top-p) filtering to logits.
    """
    top_k = min(top_k, logits.size(-1))  # Safety

    if top_k > 0:
        # Remove all tokens with a probability less than the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Always keep the first token
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def resize_embedding_preserve_weights(emb: nn.Embedding, new_num_embeddings: int, init_std: float = 0.02) -> nn.Embedding:
    old_weight = emb.weight.data
    old_n, dim = old_weight.shape
    if new_num_embeddings == old_n:
        return emb
    new_emb = nn.Embedding(new_num_embeddings, dim, device=old_weight.device, dtype=old_weight.dtype)
    new_emb.weight.data.normal_(mean=0.0, std=init_std)
    num_to_copy = min(old_n, new_num_embeddings)
    new_emb.weight.data[:num_to_copy] = old_weight[:num_to_copy]
    return new_emb


def resize_linear_out_preserve_weights(head: nn.Linear, new_out_features: int, init_std: float = 0.02) -> nn.Linear:
    old_w = head.weight.data
    in_features = head.in_features
    if new_out_features == old_w.shape[0]:
        return head
    new_head = nn.Linear(in_features, new_out_features, bias=False, device=old_w.device, dtype=old_w.dtype)
    new_head.weight.data.normal_(mean=0.0, std=init_std)
    num_to_copy = min(old_w.shape[0], new_out_features)
    new_head.weight.data[:num_to_copy] = old_w[:num_to_copy]
    return new_head


def tie_lm_head_to_embeddings(head: nn.Linear, embedding: nn.Embedding):
    assert head.out_features == embedding.num_embeddings
    assert head.in_features == embedding.embedding_dim
    head.weight = embedding.weight
    return head

