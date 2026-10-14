"""Parameter-light heads consuming only front, final-normalized Query states."""

import torch
from torch import nn


def slot_groups(count):
    if type(count) is not int or count < 7:
        raise ValueError("Slot needs at least seven queries")
    names, ratios = ("G12", "G3", "G4", "G5", "G6", "G78", "G9"), (2, 2, 2, 3, 2, 2, 3)
    products = [(count - 7) * weight for weight in ratios]
    sizes = [1 + p // sum(ratios) for p in products]
    order = sorted(range(7), key=lambda i: (-(products[i] % sum(ratios)), i))
    for i in order[:count - sum(sizes)]:
        sizes[i] += 1
    result, start = {}, 0
    for name, size in zip(names, sizes):
        result[name] = (start, start + size)
        start += size
    return result


def legal_xyxy(raw):
    values = raw.sigmoid()
    return torch.cat((values[..., :2], values[..., :2] + (1 - values[..., :2]) * values[..., 2:]), -1)


class StructuredSlotHead(nn.Module):
    def __init__(self, hidden_size, num_slot_queries):
        super().__init__()
        self.hidden_size, self.num_slot_queries = hidden_size, num_slot_queries
        self.groups = slot_groups(num_slot_queries)
        self.heads = nn.ModuleDict({q: nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, size))
                                   for q, size in {"Q1": 2, "Q2": 14, "Q3": 8, "Q4": 8, "Q5": 9, "Q6": 4, "Q9": 18}.items()})
        self.transition_norm = nn.LayerNorm(hidden_size)
        self.q7 = nn.Linear(hidden_size, 4)
        self.q8 = nn.Linear(hidden_size, 4)

    def forward(self, queries):
        if queries.ndim != 3 or queries.shape[1] < self.num_slot_queries or queries.shape[2] != self.hidden_size:
            raise ValueError("Slot requires [B,Nq,H] final-normalized Query states")
        queries = queries.to(self.q7.weight.dtype)
        pooled = {k: queries[:, start:stop].mean(1) for k, (start, stop) in self.groups.items()}
        out = {q: self.heads[q](pooled[g]) for q, g in {"Q1": "G12", "Q2": "G12", "Q3": "G3", "Q4": "G4",
                                                     "Q5": "G5", "Q6": "G6", "Q9": "G9"}.items()}
        out["Q1"] = out["Q1"].sigmoid()
        out["Q2"] = out["Q2"].reshape(-1, 2, 7)
        for q in ("Q3", "Q4"):
            out[q] = legal_xyxy(out[q].reshape(-1, 2, 4))
        out["Q5"] = out["Q5"].reshape(-1, 3, 3)
        out["Q6"] = out["Q6"].reshape(-1, 2, 2).sigmoid()
        transition = self.transition_norm(pooled["G78"])
        out.update(Q7=self.q7(transition), Q8=self.q8(transition))
        obstacle = out["Q9"].reshape(-1, 3, 6)
        out["Q9"] = torch.cat((legal_xyxy(obstacle[..., :4]), obstacle[..., 4:5].sigmoid(), obstacle[..., 5:6]), -1)
        return out


from utils.slot_config import SLOT_AUX_REGISTRY


def build_slot_head(hidden_size, num_slot_queries, config):
    config.validate()
    factory = SLOT_AUX_REGISTRY[config.slot_aux_type]
    if factory is None:
        return None
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.slot_init_seed)
        return factory(hidden_size, num_slot_queries)
