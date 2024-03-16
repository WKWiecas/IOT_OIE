# coding: utf-8
import string
from itertools import chain
from typing import List

import numpy as np
import torch

from .dataloaders import WikiDataset

ADP_ALIASES = {"ADP", "SCONJ"}


def split_adp_left(doc, detokenizer):

    if not doc.sentences:
        return "", ""

    words, i = doc.sentences[0].words, 0

    for i, word in enumerate(words):
        if word.upos not in ADP_ALIASES:
            break

    return (
        detokenizer.detokenize(map(lambda x: x.text, words[:i])),
        detokenizer.detokenize(chain(map(lambda x: x.text, words[i:]), map(lambda x: x.text, doc.sentences[1:]))),
    )


def split_adp_right(doc, detokenizer, drop_aux=False):

    if not doc.sentences:
        return "", ""

    words = doc.sentences[-1].words

    i = 0
    for i, word in reversed(list(enumerate(words))):
        if word.upos not in ADP_ALIASES:
            break

    # if drop_aux:
    #     doc.sentences[0]
    #     for i, word in reversed(list(enumerate(words))):
    #         if word.upos not in ADP_ALIASES:
    #             break

    return (
        detokenizer.detokenize(chain(map(lambda x: x.text, doc.sentences[:-1]), map(lambda x: x.text, words[: i + 1]))),
        detokenizer.detokenize(map(lambda x: x.text, words[i + 1 :])),
    )


def is_aux(doc):
    for word in chain.from_iterable(map(lambda x: x.words, doc.sentences)):
        if word.upos in ["AUX"]:
            return True
    return False


def postprocess_adp(triplets_for_texts: List[List[List]], pipeline, detokenizer):

    for triplets_for_text in triplets_for_texts:
        for triplet in triplets_for_text:

            relation, target = triplet[1][1:3]
            doc = pipeline(target)

            adp, target = split_adp_left(doc, detokenizer)

            triplet[1][1] = detokenizer.detokenize([relation, adp])
            triplet[1][2] = target


def get_best_labels_greedy(selection):   
    """Just taking argmax among 4 possible label scores"""
    selection = selection.cpu().numpy()  # [seq, 4]
    argmax_selection = np.argmax(selection, axis=1)   # [seq]
    return argmax_selection


def selection_of_triples_with_argmax(y_hat: torch.tensor, iou_prediction: torch.tensor):
    """
    :param y_hat: prediction
    iou_prediction : [batch_size, seq_len, num_detections, 1]
    """
    K = 3 
    # selecting argmax label
    amax = torch.argmax(y_hat, dim=-1)  # [batch, seq_len, num_detections]

    amax_temp = torch.zeros(size=(amax.shape[0], amax.shape[1], amax.shape[2]//K))
    # #########加一个iou筛选操作#############


    # 每K个detection选一个iou最大的
    iou_prediction = iou_prediction.squeeze(-1)  # [batch_size, seq_len, num_detections]
    for i in range(iou_prediction.shape[0]):
        for j in range(iou_prediction.shape[1]):
            for detection_k in range(iou_prediction.shape[-1] // K):
                iou_prediction_K = iou_prediction[i,j, K*detection_k : K*detection_k + K]    # [K]
                iou_arg_max = torch.argmax(iou_prediction_K, dim=-1)   # 1
                amax_temp[i][j][detection_k] = amax[i, j, K*detection_k : K*detection_k + K][iou_arg_max]

    amax = amax_temp

    # checking if equals any of the SRT labels (note: broadcasting; [2] -> [False True False])
    checking = amax.unsqueeze(-1) == torch.tensor(
        [1, 2, 3], device=amax.device
    )  # boolean [batch, seq_len, num_detections, 3]

    # aggregation along sequences
    # 表示在sequence这一维度上有，则是true
    any_has_true = torch.any(checking, dim=-3)  # [batch, num_detections, 3]

    # checking if all of the three labels are present     
    # 在最后一维上要求当前S R O 都有
    all_of_srt_are_present = torch.all(any_has_true, dim=-1)  # [batch, num_detections]

    # getting the ids of the detections where this holds
    nonzeros_item_ids, nonzeros_rel_ids = all_of_srt_are_present.nonzero(as_tuple=True)   # 分别记录了横纵坐标    是针对batch内的所有样本做的

    return nonzeros_item_ids.cpu().numpy(), nonzeros_rel_ids.cpu().numpy()


def selection_of_triples(y_hat, threshold: float = 0.1, strict_all_three=True):
    """
        Softmaxed prediction -- to the lists of significant detections per data point in a batch
    :param y_hat: prediction after softmax
    :param threshold: filtering threshold
    :param strict_all_three: True if at least one label from each of S,R,T is present
    """
    y_hat = torch.softmax(y_hat, dim=-1)
    mask = y_hat > threshold

    # sequences have at least THRESHOLD probability
    mask_all_sequences_are_meaningful = torch.any(mask[:, :, :, WikiDataset.NO + 1 :], dim=-3)

    # detections that have at least one significant occurrence in a seq of each of the three: S, R, T
    mask_all_three_parts_are_there = (
        torch.all(mask_all_sequences_are_meaningful, dim=-1)
        if strict_all_three
        else torch.any(mask_all_sequences_are_meaningful, dim=-1)
    )

    # a list of such detections per item_id (number of data point in the batch)
    nonzeros_item_ids, nonzeros_rel_ids = mask_all_three_parts_are_there.nonzero(as_tuple=True)

    return nonzeros_item_ids.cpu().numpy(), nonzeros_rel_ids.cpu().numpy()


def strip_bs(text: str) -> str:
    """Removing punctuation and whitespace from bowth sides of the text"""
    return text.strip(string.punctuation).strip()


def spans2triples(labels, text, offsets_mapping_item, tokens):
    """
        Labels to text
    :param labels: NSRT = 0,1,2,3
    :param text: a single paragraph
    :param offsets_mapping_item: spans yielded by tokenizer
    """

    # let us believe the tokens of the same tag follow each other without breaks
    result = [None, None, None, None]
    prev_label = 0

    for label, token, span in zip(labels, tokens, offsets_mapping_item):

        if token.startswith("##"):
            label = prev_label

        if result[label] is not None:
            result[label] = (result[label][0], span[1])       # 直接取最长的那一串span吗？？？？这里实在是没看懂。。。
        else:
            result[label] = span

        prev_label = label

    return [strip_bs(text[sp[0] : sp[1]]) if sp is not None else "" for sp in result[1:]]


def prediction2triples(prediction, texts, offsets_mapping, tokenized, iou_prediction):

    with torch.no_grad():
        y_hat = prediction   # [bs, seq, n_detection, 4]
        nonzeros_item_ids, nonzeros_rel_ids = selection_of_triples_with_argmax(y_hat, iou_prediction)     # nonzeros_item_ids一维的list，记录了所有可能的结果如113
        triplets = [[] for _ in range(len(texts))]

        for item_id, rel_id in zip(nonzeros_item_ids, nonzeros_rel_ids):      # zip(第几个样本, 第几个detector)
            pred_labels = get_best_labels_greedy(y_hat[item_id, :, rel_id, :])   # 输出每个token的分类，输出大小为[seq]

            # todo: uncomment to do minor alignment
            # pred_labels = postprocesing_labels(pred_labels)
            # print(pred_labels)
            # print(texts[item_id], offsets_mapping[item_id], tokenized[item_id].tokens)

            triples = spans2triples(pred_labels, texts[item_id], offsets_mapping[item_id], tokenized[item_id].tokens)
            triplets[item_id].append([rel_id.item(), triples])

    return triplets


def pprint_triplets(texts: List[str], triplets: List[List[str]], **kwargs):
    for paragraph, t in zip(texts, triplets):
        print(paragraph, **kwargs)
        for rel_id, triplet in t:
            print("[%3d]" % rel_id, triplet, **kwargs)
