# coding: utf-8
import io
from collections import OrderedDict
from typing import List, Optional

import pytorch_lightning as pl
import stanza
import torch
from lapsolver import solve_dense
from nltk.tokenize.treebank import TreebankWordDetokenizer
from omegaconf import DictConfig
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertModel

from .apply import postprocess_adp, pprint_triplets, prediction2triples
from .dataloaders import SyntaxFeatures, make_syntax_features
from .feature_preparation import syntax_based_features_for_bpe
from .tags import UD_DEPREL, UPOS_TAGS_ALL
import time

# OT包引用
from OT.kernel import OTKernel
from OT.my_ot_layer import OTLayer

from torch.nn.utils.rnn import pad_sequence


class TransposeLayer(nn.Module):
    """A helper class for transposing the tensor"""

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, input: torch.Tensor):
        return input.transpose(*self.args)


class SinkhornDistance(torch.nn.Module):
    r"""
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu).cuda()
        v = torch.ones_like(nu).cuda()

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * \
                (torch.log(
                    nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * \
                (torch.log(
                    mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(
            self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(
            pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps



class TripletsExtractor(pl.LightningModule):
    """Base class for all single-shot models extracting multiple triplets"""

    def __init__(self, model_cfg: DictConfig, opt_cfg: DictConfig, scheduler_cfg: DictConfig):
        super().__init__()

        self.MAX_GT_NUM = {}
        self.save_hyperparameters()
        self.model_cfg, self.opt_cfg = model_cfg, opt_cfg

        # print('self.model_cfg', self.model_cfg)

        # Init side tools (stanza & detokenizer)
        self.lang = model_cfg.lang
        self.postprocess_adp = model_cfg.postprocess_adp
        self.use_syntax_features = model_cfg.use_syntax_features
        self.init_tools()

        self.seed = model_cfg.seed
        self.example_texts = list(model_cfg.viz_sentences)
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer, cache_dir=model_cfg.cache_dir)
        self.pretrained_encoder = AutoModelForMaskedLM.from_pretrained(
            model_cfg.pretrained_encoder, cache_dir=model_cfg.cache_dir
        ).base_model.cuda()

        in_size = self.pretrained_encoder.config.hidden_size
        hid_size = in_size
        out_dim = model_cfg.num_detections * model_cfg.n_classes

        if model_cfg.use_syntax_features:
            in_size = in_size + 2 * model_cfg.stanza_emb_size
            self.pos_emb = nn.Embedding(len(UPOS_TAGS_ALL) + 1, model_cfg.stanza_emb_size)
            self.deprel_emb = nn.Embedding(len(UD_DEPREL) + 1, model_cfg.stanza_emb_size)

        self.logits = nn.Sequential(
            nn.Linear(in_size, hid_size),
            # TransposeLayer(1, 2),
            # nn.Conv1d(model_cfg.pretrained_emb_size, hid_size, kernel_size=5, padding=2),
            # TransposeLayer(1, 2),
            # nn.ReLU(),
            nn.LayerNorm(hid_size),
            TransposeLayer(0, 1),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hid_size, model_cfg.n_classes, hid_size), num_layers=model_cfg.num_layers     # TransformerEncoderLayer参数是（input_dim=hid_size, nhead=n_classes, output_dim=hid_size）
            ),
            TransposeLayer(0, 1),
            nn.ReLU(),
            nn.Linear(hid_size, out_dim),   # out_dim比较大，输出变成了[bs, seq_len, num_detections * n_classes]  其中num_detections 默认为20
        )

        
        self.top_candidates = model_cfg.top_candidates # OT的针对每个样本取topk
        self.sinkhorn = SinkhornDistance(eps=0.1,
                                    max_iter=self.model_cfg.ot_iteration)

        self.compute_crossentropy_with_logits = nn.CrossEntropyLoss(
            weight=torch.tensor(list(model_cfg.class_weights), dtype=torch.float), reduction="none"
        )

        name, opt_cfg = opt_cfg.name, dict(opt_cfg)
        del opt_cfg["name"]
        self.opt = getattr(torch.optim, name)(self.parameters(), **opt_cfg)

        name, scheduler_cfg = scheduler_cfg.name, dict(scheduler_cfg)
        del scheduler_cfg["name"]
        self.scheduler = getattr(torch.optim.lr_scheduler, name)(self.opt, **scheduler_cfg)

        import torchmetrics
        self.metrics = {}
        for stage in ("train", "val", "test"):
            self.metrics[self.get_metric_name("f1_score", stage)] = torchmetrics.F1Score(
                model_cfg.n_classes, average="macro"
            ).cuda()
            self.metrics[
                self.get_metric_name("precision", stage)
            ] = torchmetrics.Precision(model_cfg.n_classes, average="macro").cuda()
            self.metrics[self.get_metric_name("recall", stage)] = torchmetrics.Recall(
                model_cfg.n_classes, average="macro"
            ).cuda()


    def init_tools(self, cache_dir: str = None):

        if self.postprocess_adp:
            self.detokenizer = TreebankWordDetokenizer()

        if self.postprocess_adp or self.use_syntax_features:
            if not cache_dir:
                cache_dir = self.model_cfg.cache_dir

            stanza.download(self.lang, model_dir=cache_dir)
            self.stanza_pipeline = stanza.Pipeline(lang=self.lang, processors="tokenize,mwt,pos", dir=cache_dir)

    @staticmethod
    def get_metric_name(name, stage, tag: str = ""):
        if not tag:
            return f"{stage}_{name}"
        return f"{stage}_{name}_{tag}"

    def _get_stage_metrics(self, stage):
        return {key: value for key, value in self.metrics.items() if key.startswith(stage)}

    def forward(self, encoder_inputs, syntax_features: Optional[SyntaxFeatures] = None, labels=None, matching="iou", disable_bg=True, eps=1e-8, training_flag=False):
        # BERT embeddings
        encoder_outputs = self.pretrained_encoder(**encoder_inputs)  # [batch_size, seq_len, encoder_hid_size]

        # last hidden state goes into the main head
        main_head_input = encoder_outputs.last_hidden_state

        if self.use_syntax_features:
            pos_emb = self.pos_emb(syntax_features.pos_tags)
            deprel_emb = self.deprel_emb(syntax_features.deprel_tags)
            main_head_input = torch.cat([main_head_input, pos_emb, deprel_emb], -1)

        # Filter everything that is below threshold
        logits = self.logits(main_head_input)  #  加了一层transformer,输出大小为[batch_size, seq_len, n_classes * num_detections]

        batch_size, seq_len, *_ = logits.shape
        logits = logits.view(
            *logits.shape[:-1], -1, self.model_cfg.n_classes
        )  # 输出变成 [batch_size, seq_len, num_detections, n_classes]

        # 测试时直接返回logits
        if not training_flag:
            return logits
        
        # 训练的话要计算iou
        else:
            # normalizing predictions
            detached_probas = torch.softmax(logits, dim=-1).detach()    # 给预测值归一化 [batch_size, seq_len, num_detections, n_classes]

            effective_labels = labels
            if disable_bg:
                # removing background for IoU computation  
                # 默认class第一个是背景O
                detached_probas = detached_probas[:, :, :, 1:]        # [bs, seq_len, num_detections, n_classes-1]
                effective_labels = effective_labels[:, :, :, 1:]      # [bs, seq_len, 14, n_classes-1]  *****14是ground truth里面所有的输出可能，每个batch不同******

            # m -- how many detectors we decided to have
            # n -- how many actual relations we are supposed to extract
            intersection = torch.einsum("ijmk,ijnk->imn", detached_probas, effective_labels.to(torch.float))      # [bs, num_detections, 14]
            sum_probas = torch.sum(detached_probas, dim=(1, -1)).unsqueeze(-1)    # [bs, num_detections, 1]
            sum_labels = torch.sum(effective_labels, dim=(1, -1)).unsqueeze(-2)   # [bs, 1, 14]
            if matching == "iou":
                # Inclusion–exclusion principle
                union = sum_probas + sum_labels - intersection       # [bs, num_detections, 14]
            elif matching == "dice":
                union = sum_probas + sum_labels
            elif matching == "dice_squared":
                sum_probas = torch.sum(detached_probas**2, dim=(1, -1)).unsqueeze(-1)
                union = sum_probas + sum_labels
            else:
                raise NotImplementedError

            # IoU scores for every pair of detection-vs-actual_relation for every element of the batch
            # 计算IoU
            matching_metrics = intersection / (union + eps)  # [batch_size, num_filtered_detections, n_relations]  # 14?
            matching_metrics = matching_metrics.cuda()

            batch_size = labels.shape[0]
            num_gt = labels.shape[-2]                     # e.g., 14
            num_anchor = detached_probas.shape[-2]        # 20

            if num_gt > 20:
                if num_gt not in self.MAX_GT_NUM.keys():
                    self.MAX_GT_NUM[num_gt] = 1
                else:
                    self.MAX_GT_NUM[num_gt] += 1
                print('MAX_GT_NUM', self.MAX_GT_NUM)

            batch_iou = []
            matched_labels, labels = torch.zeros_like(logits, dtype=torch.int).cuda(), labels.to(torch.int).cuda()

            # bg_labels = torch.zeros([labels.shape[0], labels.shape[1], num_anchor-num_gt, labels.shape[3]]).to(labels.device)   # [batch, seq, 6, n_classes]
            bg_labels = torch.zeros([labels.shape[0], labels.shape[1], 1, labels.shape[3]]).to(labels.device)

            labels = torch.cat([labels, bg_labels], dim=-2)  # 拼上背景的全0label，[batch, seq, 15, n_classes]       # lsoie [batch, seq, 30, n_classes]

            # 对每个batch做ot
            for i, iou_scores in enumerate(matching_metrics):     # iou_scores [20, 14]   
                
                iou_scores = iou_scores.permute(1,0)  # [14, 20]   前面是gt，后面是pred

                if not self.model_cfg.dynamic_k:
                    '''k写死的方法'''
                    topk_ious, _ = torch.topk(iou_scores, self.top_candidates, dim=1)   # 对每一个gt框求topk个pred结果   [14, 4]
                    topk_iou_sum_list = torch.clamp(topk_ious.sum(1).int(), min=1).float()
                
                else:
                    '''动态k 每个三元组用不同的k batch wise?      sample wise?      tuple wise?'''
                    iousum_pre_gt = torch.sum(iou_scores, 1)  # [14]

                    if self.model_cfg.iou_ceil:
                        iousum_pre_gt = iousum_pre_gt.ceil()  # 向上取整 [14]  
                    else:
                        iousum_pre_gt = iousum_pre_gt.floor()  # 向上取整 [14]  
                    
                    # 给k设一个上下限 一个真值最多配1-3个框
                    iousum_pre_gt = torch.clamp(iousum_pre_gt.int(), min=1, max=self.model_cfg.topk_max)
                    iousum_pre_gt = iousum_pre_gt.cpu().numpy().tolist()

                    topk_iou_sum_list = []
                    for gt_i, iou_score_per_gt in enumerate(iou_scores):    # iou_score_per_gt [20]
                        topk_iou_per_gt, _ = torch.topk(iou_score_per_gt, iousum_pre_gt[gt_i])     # 每个真值取不同的topk   [topk]
                        topk_iou_per_gt = topk_iou_per_gt.cuda()
                        topk_iou_per_ge_sum = torch.sum(topk_iou_per_gt)   # [1]
                        topk_iou_sum_list.append(topk_iou_per_ge_sum)
                    topk_iou_sum_list = torch.Tensor(topk_iou_sum_list).int().cuda()
                    topk_iou_sum_list = torch.clamp(topk_iou_sum_list, min=1).float()   #[14]

                mu = iou_scores.new_ones(num_gt+1)    # 全1的大小为[14+1]的矩阵, +1是考虑背景
                mu[:-1] = topk_iou_sum_list
                mu[-1] = num_anchor - mu[:-1].sum()    # 最后一个background需要M-N*K个资源
                
                nu = iou_scores.new_ones(num_anchor)     # [20]
                
                bg_loss = torch.zeros([1, iou_scores.shape[1]]).to(iou_scores.device)  # 背景的cost，默认是0   [1, 20]
                loss = torch.cat([iou_scores, bg_loss], dim=0).cuda()    # [15, 20]
                loss = (-1) * loss  # [15, 20]

                # Solving Optimal-Transportation-Plan pi via Sinkhorn-Iteration.
                _, pi = self.sinkhorn(mu, nu, loss)        # pi [15, 20]
                
                # Rescale pi so that the max pi for each gt equals to 1.
                rescale_factor, _ = pi.max(dim=1)
                pi = pi / rescale_factor.unsqueeze(1)

                # 获得了最合适的分配策略
                max_assigned_units, matched_gt_inds = torch.max(pi, dim=0)   # [20]

                # iou用0补齐，方便观察
                if num_anchor > num_gt:
                    iou_scores_zeros = torch.zeros(num_anchor-num_gt, iou_scores.shape[1]).to(iou_scores.device)
                    iou_scores = torch.cat([iou_scores, iou_scores_zeros], dim=0) 

                # 每个预测框都和一个真值计算标签。要加个背景label
                for pred_i, matched_gt_idx in enumerate(matched_gt_inds):
                    matched_labels[i, :, pred_i] = labels[i, :, matched_gt_idx]
                    if matched_gt_idx != labels.shape[-2]-1:
                        batch_iou.append(iou_scores[matched_gt_idx, pred_i].mean())
                    else:
                        batch_iou.append(0)
                
            return logits, matched_labels, sum(batch_iou) / len(batch_iou)


    def predict(self, texts: List[str], calc_confidence=False, **kwargs):
        if self.training:
            self.eval()

        self.model_cfg.join_is = True
        
        if self.model_cfg.join_is:
            texts = [text + " [is] [of] [from]" for text in texts]

        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
            # added June 3rd for running on Russian
            truncation=True,
            max_length=self.model_cfg.hid_size,
        ).to(self.device)

        syntax_features = None
        if self.use_syntax_features:
            batch = [
                list(
                    map(
                        lambda items: [item if item is not None else 0 for item in items],
                        syntax_based_features_for_bpe(text, self.stanza_pipeline, self.tokenizer),
                    )
                )
                for text in texts
            ]
            syntax_features = make_syntax_features(*zip(*batch), device=self.device)

        offsets_mapping = tokenized["offset_mapping"].cpu().numpy()
        del tokenized["offset_mapping"]  # otherwise -- error
        special_tokens = set(self.tokenizer.special_tokens_map.values())

        with torch.no_grad():
            prediction = self.forward(tokenized, syntax_features)    # [batch_size, seq_len, num_detections, n_classes]
            triplets = prediction2triples(
                prediction=prediction,
                texts=texts,
                offsets_mapping=offsets_mapping,
                tokenized=tokenized,
            )

        if self.postprocess_adp:
            postprocess_adp(triplets, self.stanza_pipeline, self.detokenizer)

        return triplets

    def compute_loss(self, batch, calc_metrics=True, stage=None):
        (
            encoder_inputs,
            labels_one_hot,
            syntax_features,
        ) = batch  # y.shape == [batch_size, seq_len, n_relations, n_classes]

        matched_logits, matched_labels_one_hot, avg_iou = self.forward(encoder_inputs, syntax_features, labels_one_hot, 
                                                                       matching=self.model_cfg.matching, disable_bg=self.model_cfg.disable_bg,
                                                                       training_flag=True)   # 把特征维通过全连接拆开了，输出是 [batch_size, seq_len, num_detections, n_classes]

        if self.model_cfg.focal_gamma != 0:
            matched_probas = torch.softmax(matched_logits, -1).max(-1)[0].view(-1)        # 需要变成      [bs, num_detection, n_classes, seq, seq]
        matched_logits = matched_logits.view(-1, matched_logits.size(-1)).cuda()  #     matched_logits: [batch_size, seq_len, num_detections, n_classes]
        matched_labels = matched_labels_one_hot.argmax(-1).view(-1).cuda()  #           matched_labels_one_hot:[batch_size, seq_len, num_detections, n_classes]

        
        loss = self.compute_crossentropy_with_logits(matched_logits, matched_labels)   # label部分用0补齐了，计算loss时等于0
        if self.model_cfg.focal_gamma != 0:
            loss *= (1 - matched_probas) ** self.model_cfg.focal_gamma
        loss = loss.mean()

        metrics = OrderedDict()
        if calc_metrics:
            assert stage

            metrics[self.get_metric_name("loss", stage)] = loss.item()
            metrics[self.get_metric_name("avg_iou", stage)] = avg_iou.item()

            matched_preds = matched_logits.argmax(-1)    # 对整个batch拉平，然后计算当前label是否预测对了
            matched_preds = matched_preds.cuda()
            matched_labels = matched_labels.cuda()
            for metric_name, metric_fn in self._get_stage_metrics(stage).items():
                metrics[metric_name] = metric_fn(matched_preds, matched_labels)

        return matched_logits, loss, metrics

    @staticmethod
    def _set_req_grad(module: nn.Module, value: bool):
        for param in module.parameters():
            param.requires_grad = value

    def on_epoch_start(self):
        if self.current_epoch == self.model_cfg.unfreeze_epoch:
            n_layers = len(self.pretrained_encoder.encoder.layer)
            print(
                f"Unfreezing layers starting after {n_layers - self.model_cfg.unfreeze_layers_from_top} "
                f"of {n_layers} on epoch {self.current_epoch}"
            )
            for layer in self.pretrained_encoder.encoder.layer[n_layers - self.model_cfg.unfreeze_layers_from_top :]:
                self._set_req_grad(layer, True)

    def on_fit_start(self):
        pl.seed_everything(self.seed)
        self._set_req_grad(self.pretrained_encoder, False)

    def _log_metrics(self, metrics: dict, postfix: str = "_batch"):
        for key, value in metrics.items():
            self.log(key + postfix, value)

    def _log_epoch_metrics(self, stage, postfix: str = "_epoch"):
        for metric_name, metric_fn in self._get_stage_metrics(stage).items():
            print(metric_name + postfix, metric_fn.compute())
            self.log(metric_name + postfix, metric_fn.compute(), on_step=False, on_epoch=True)

    def log_example_prediction(self):
        triplets = self.predict(self.example_texts)
        buf = io.StringIO()
        pprint_triplets(self.example_texts, triplets, end="  \n", file=buf)
        self.logger.experiment.add_text("predictions", buf.getvalue())

    def training_step(self, batch, batch_idx):
        _, loss, metrics = self.compute_loss(batch, stage="train")
        self.scheduler.step()
        self._log_metrics(metrics)
        return loss

    def on_train_epoch_end(self, *args):
        self._log_epoch_metrics("train")

    def validation_step(self, batch, batch_idx):
        _, _, metrics = self.compute_loss(batch, stage="val")
        self._log_metrics(metrics)

    def on_validation_epoch_end(self):
        self.log_example_prediction()
        self._log_epoch_metrics("val")

    def test_step(self, batch, batch_idx):
        _, _, metrics = self.compute_loss(batch, stage="test")
        self._log_metrics(metrics)

    def on_test_epoch_end(self):
        self.log_example_prediction()
        self._log_epoch_metrics("test")

    def configure_optimizers(self):
        return self.opt


class TripletsExtractorBERTOnly(TripletsExtractor):
    def __init__(self, model_cfg: DictConfig, opt_cfg: DictConfig, scheduler_cfg: DictConfig):
        super().__init__(model_cfg, opt_cfg, scheduler_cfg)
        self.logits = nn.Linear(
            self.pretrained_encoder.config.hidden_size, model_cfg.num_detections * model_cfg.n_classes
        )
