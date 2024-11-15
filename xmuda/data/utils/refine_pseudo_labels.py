import torch


def refine_pseudo_labels(probs, pseudo_label, ignore_label=-100):
    probs, pseudo_label = torch.tensor(probs), torch.tensor(pseudo_label)
    for cls_idx in pseudo_label.unique():
        curr_idx = pseudo_label == cls_idx
        curr_idx = curr_idx.nonzero().squeeze(1)
        thresh = probs[curr_idx].median()
        thresh = min(thresh, 0.9)
        ignore_idx = curr_idx[probs[curr_idx] < thresh]
        pseudo_label[ignore_idx] = ignore_label
    return pseudo_label.numpy()
