# coding: utf-8

import json
from collections import Sequence, defaultdict, namedtuple
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

SyntaxFeatures = namedtuple("SyntaxFeatures", ["heads", "pos_tags", "deprel_tags"])


def find_max_lens(nested_lists, lens, depth=0):

    if not isinstance(nested_lists, Sequence):
        return

    for item in nested_lists:
        if isinstance(item, Sequence):
            lens[depth] = max(lens[depth], len(item))
            find_max_lens(item, lens, depth + 1)


def pad_nested_lists_(nested_lists, lens, fill_value=0):

    for item in nested_lists:
        if isinstance(item, list):
            item.extend(np.full([lens[0] - len(item), *lens[1:]], fill_value).tolist())
            pad_nested_lists_(item, lens[1:])


def pad_nested_lists(nested_lists: List):

    lens = defaultdict(int)
    find_max_lens(nested_lists, lens)

    _, lens = zip(*sorted(lens.items()))
    pad_nested_lists_(list(nested_lists), lens)

    return torch.tensor(nested_lists)


class WikiDataset(Dataset):
    """
    Reading all the relations and the texts from the supplied JSON-file
    and keeping them in RAM
    """

    NO, S, R, T = 0, 1, 2, 3

    def __init__(
        self,
        path: str,
        tokenizer_name: str,
        use_syntax_features: bool = False,
        combine_multiple_sentences: bool = False,
        for_debug=False # 测试用
    ):

        # texts -- raw texts
        # relations -- triples in a string form
        # indices -- triples of char-spans for each of the relations
        with open(path) as fp:
            if not for_debug:
                self.json_dict = json.load(fp)    # 包括indices，relations,texts三个部分
            else:
                self.json_dict = json.load(fp)[0:100]    # 包括indices，relations,texts三个部分
        self.tokenizer_name = tokenizer_name
        self.use_syntax_features = use_syntax_features
        self.combine_multiple_sentences = combine_multiple_sentences

    def __getitem__(self, i):
        if self.use_syntax_features:
            return (
                self.json_dict["texts"][i],
                self.json_dict["indices"][i],
                [item if item is not None else 0 for item in self.json_dict["heads"][self.tokenizer_name][i]],
                self.json_dict["pos_tags"][self.tokenizer_name][i],
                self.json_dict["deprel_tags"][self.tokenizer_name][i],
            )
        return self.json_dict["texts"][i], self.json_dict["indices"][i], None, None, None

    def __len__(self):
        return len(self.json_dict["texts"])


def pad_lists(batch: List[List], device: torch.device = torch.device("cpu")):
    return pad_sequence(list(map(lambda x: torch.tensor(x, device=device), batch)), batch_first=True)


def make_syntax_features(
    heads: List[List], pos_tags: List[List], deprel_tags: List[List], device: torch.device = torch.device("cpu")
) -> SyntaxFeatures:
    return SyntaxFeatures(*map(lambda x: pad_lists(x, device), (heads, pos_tags, deprel_tags)))


@dataclass
class CollateFn:

    tokenizer: Callable
    use_syntax_features: bool = False
    word_dropout: float = False
    multiple_spans: bool = False
    verbose: bool = False

    def __call__(self, batch: List) -> (torch.Tensor, torch.Tensor, Optional[SyntaxFeatures]):

        texts, indices, heads, pos_tags, deprel_tags = zip(*batch)
        tokenized = self.tokenizer(list(texts), return_tensors="pt", padding=True, return_offsets_mapping=True, max_length=512, truncation=True)  # 我加了一个truncation操作，不然有的会很长

        offset_mapping = tokenized["offset_mapping"]
        batch_size, seq_len, _ = offset_mapping.shape

        # if seq_len > 512:
            # print('list(texts)', list(texts))

        if self.multiple_spans:
            offset_mapping = offset_mapping.view(batch_size, seq_len, 1, 1, 1, 2)

            # (batch_size, n_rel, n_relations, 2)  n_relations是ground truth补齐后的个数  # bs, 14, 3, 4, 2  
            indices = pad_nested_lists(indices)   # 往一个batch里三元组最多的补齐
            # 错的 (batch_size, n_triple, 3, 3, num_spans, 2)                                          3是三元组   4是什么？
            indices = indices.view(batch_size, 1, -1, 3, indices.shape[-2], 2)      # 输出是[bs, 1, n_triple, 3, 4, 2]

            # using original spans and token spans (offset_mapping) for masks
            masks = torch.any(
                (offset_mapping[:, :, :, :, :, 0] >= indices[:, :, :, :, :, 0])
                & (offset_mapping[:, :, :, :, :, 1] <= indices[:, :, :, :, :, 1])   # 前两个是在找满足要求的三元组位置
                & torch.any(indices != 0, -1)    # 去掉indices补零的
                & torch.any(offset_mapping != 0, -1),    # 去掉offset_mapping补零的
                -1,
            )  # [batch_size, seq_len, n_relations, n_classes-1]
        else:
            offset_mapping = offset_mapping.view(batch_size, seq_len, 1, 1, 2)
            # max_len = max(map(len, indices))

            # converting original spans into pytorch format
            # each item is a triple of spans: [source, relation, target], e.g. [[0, 12], [14, 29], [45, 90]]
            # manual padding is done below
            indices = pad_nested_lists(indices)
            # indices = torch.tensor([item + [[[0, 0]] * 3] * (max_len - len(item)) for item in indices])
            # [32, 10, 3, 2] -> [32, 1, 10, 3, 2]
            indices = indices.view(batch_size, 1, -1, 3, 2)

            # using original spans and token spans (offset_mapping) for masks
            masks = (
                (offset_mapping[:, :, :, :, 0] >= indices[:, :, :, :, 0])
                & (offset_mapping[:, :, :, :, 1] <= indices[:, :, :, :, 1])
                & torch.any(indices != 0, -1)
                & torch.any(offset_mapping != 0, -1)
            )  # [batch_size, seq_len, n_relations, n_classes]

        # masks are in order: no_relation, source, relation, target; right?   # logical_not 相当于取反，第一维得到no_relation
        masks = torch.cat([torch.logical_not(torch.sum(masks, -1, keepdim=True)), masks], -1)  # 输出是[bs, seq, n_relations, 4]
        
        # try:

        if self.verbose:
            mislabeled_triplets = torch.sum(masks.sum(-1) > 1)
            print("mislabeled_triplets:", mislabeled_triplets)

        del tokenized["offset_mapping"]

        syntax_features = None
        if self.use_syntax_features:
            syntax_features = make_syntax_features(heads, pos_tags, deprel_tags)

        if self.word_dropout:
            dropout_mask = torch.rand(seq_len) >= self.word_dropout
            dropout_mask[0] = True
            dropout_mask = dropout_mask | torch.any(tokenized["input_ids"] == self.tokenizer.sep_token_id, dim=0)

            for key, value in tokenized.items():
                tokenized[key] = value[:, dropout_mask]
            masks = masks[:, dropout_mask]

            if syntax_features:
                syntax_features.heads = syntax_features.heads[:, dropout_mask]
                syntax_features.deprel_tags = syntax_features.deprel_tags[:, dropout_mask]
                syntax_features.pos_tags = syntax_features.pos_tags[:, dropout_mask]

        return tokenized, masks, syntax_features    # 这里输出的mask是实际的label


if __name__ == "__main__":
    from torch.utils.data.dataset import random_split

    dataset = WikiDataset("../../data/wikidata/sentences.json")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    val_size = int(len(dataset) * 0.05)

    # torch utils random split
    train, val = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train, batch_size=1, collate_fn=CollateFn(tokenizer))

    for i in train_loader:
        print(i)
        break
