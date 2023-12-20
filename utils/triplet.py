import torch


def negative_only_triplet_loss(labels, embeddings, k, margin=0, margin_type='soft'):
    """Variant of the triplet loss, computed only to separate the hardest negatives.

    See `batch_hard_triplet_loss` for details.

    Args:
        labels: labels of the batch, of shape (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        k: number of negatives to consider
        margin: margin for triplet loss
        margin_type: 'soft' or 'hard'. If 'soft', the loss is `log(1 + exp(positives - negatives + margin))`.
            If 'hard', the loss is `max(0, positives - negatives + margin)`.

    Returns:
        torch.Tensor: scalar tensor containing the triplet loss
    """
    k = min(k, labels.shape[0])

    # Get the pairwise distance matrix
    pairwise_dist = (embeddings.unsqueeze(0) - embeddings.unsqueeze(1)).pow(2).sum(2)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1)).float()

    # We add inf in each row to the positives
    anchor_negative_dist = pairwise_dist
    anchor_negative_dist[mask_anchor_positive.bool()] = float('inf')

    # shape (batch_size,)
    hardest_negative_dist = torch.topk(anchor_negative_dist, k=k, dim=1, largest=False)[0]
    mask = hardest_negative_dist != float('inf')

    dneg = hardest_negative_dist[mask]

    if dneg.shape[0] == 0:
        return None

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    if margin_type == 'soft':
        loss = torch.log1p(torch.exp(- dneg + float(margin)))
    else:
        loss = torch.clamp(- dneg + float(margin), min=0.0)

    # Get thanchor_negative_diste true loss value
    loss = torch.mean(loss)

    return loss


def batch_hard_triplet_loss(labels, embeddings, k, margin=0, margin_type='soft'):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, get the hardest positive and hardest negative to compute the triplet loss.

    Args:
        labels: labels of the batch, of shape (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        k: number of negatives to consider
        margin: margin for triplet loss
        margin_type: 'soft' or 'hard'. If 'soft', the loss is `log(1 + exp(positives - negatives + margin))`.
            If 'hard', the loss is `max(0, positives - negatives + margin)`.

    Returns:
        torch.Tensor: scalar tensor containing the triplet loss
    """
    k = min(k, labels.shape[0])

    # Get the pairwise distance matrix
    pairwise_dist = (embeddings.unsqueeze(0) - embeddings.unsqueeze(1)).pow(2).sum(2)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1)).float()

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive * pairwise_dist

    # shape (batch_size, 1)
    hardest_positive_dist = torch.topk(anchor_positive_dist, k=k, dim=1, largest=True)[0]

    # We add inf in each row to the positives
    anchor_negative_dist = pairwise_dist
    anchor_negative_dist[mask_anchor_positive.bool()] = float('inf')

    # shape (batch_size,)
    hardest_negative_dist = torch.topk(anchor_negative_dist, k=k, dim=1, largest=False)[0]
    mask = hardest_negative_dist != float('inf')

    dpos = hardest_positive_dist[mask]
    dneg = hardest_negative_dist[mask]

    if dpos.shape[0] == 0 or dneg.shape[0] == 0:
        return None

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    if margin_type == 'soft':
        loss = torch.log1p(torch.exp(dpos - dneg + float(margin)))
    else:
        loss = torch.clamp(dpos - dneg + float(margin), min=0.0)

    # Get thanchor_negative_diste true loss value
    loss = torch.mean(loss)

    return loss
