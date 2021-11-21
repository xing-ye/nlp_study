#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import torch
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertModel, BertConfig
from transformers import AdamW
from transformers import BertTokenizer
# from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.cuda.amp import autocast, GradScaler
from torch.optim.optimizer import Optimizer
import math
# import Levenshtein
from pypinyin import lazy_pinyin
import jieba


# 设置参数及文件路径
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 程序可调用的GPU的ID
max_seq_length = 60  # 输入文本最大长度
learning_rate = 2e-5  # 模型学习率
num_epochs = 7  # 训练最大迭代次数
batch_size = 160  # 训练时每个batch中的样本数
patience = 5  # 早停轮数
file_name = 'baseline'  # 指定输出文件的名字
model_name_or_path = './pretrain_models/ernie_grame/'  # 预训练模型权重载入路径
train_input = './data/train/'  # 完成预处理的训练集载入路径
test_input = './data/test/'  # 完成预处理的测试集载入路径
random_seed = 42  # 随机种子

# 固定随机种子
def seed_everything(seed=random_seed):
    '''
    固定随机种子
    :param random_seed: 随机种子数目
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()

# 创建一个logger
file_path = './log/'
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_model1.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

#定义原始数据类型
class InputExample(object):
    def __init__(self, s1, s2, label=None):
        self.s1 = s1
        self.s2 = s2
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 choices_features,
                 label

                 ):
        _, input_ids, input_mask, segment_ids = choices_features[0]
        self.choices_features = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids
        }
        self.label = label

#读取某一文件数据
def read_data(file_name):
    examples = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                line = line.split('\t')
                examples.append(InputExample(s1=line[0], s2=line[1], label=int(line[2]) if len(line) == 3 else None))
    return examples

#读取多个文件数据并合并
def read_examples(dir, split='train'):
    examples = []
    for path in os.listdir(dir):
        if split == 'train':
            for file_name in os.listdir(dir + path):
                example = read_data(os.path.join(dir+path, file_name))
                examples.extend(example)
        else:
            example = read_data(os.path.join(dir, path))
            examples.extend(example)
    return examples

#将序列对原地截断为最大长度。
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """T这是一个简单的启发式方法，它总是一次一个标记截断较长的序列。
    这比从每个中截断相同百分比的标记更有意义，因为如果一个序列非常短，
    那么被截断的每个标记可能比更长的序列包含更多的信息。"""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

# 将文本输入样例，转换为数字特征，用于模型计算
def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):

    features = []
    for example_index, example in enumerate(examples):

        s1 = tokenizer.tokenize(example.s1)
        s2 = tokenizer.tokenize(example.s2)
        _truncate_seq_pair(s1, s2, max_seq_length)

        choices_features = []

        tokens = ["[CLS]"] + s1 + ["[SEP]"] + s2 + ["[SEP]"]
        segment_ids = [0] * (len(s1) + 2) + [1] * (len(s2) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids) + 3
        input_ids += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        segment_ids += ([0] * padding_length)
        choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        if example_index < 1 and is_training:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example_index))
            logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
            logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            logger.info("input_mask: {}".format(len(input_mask)))
            logger.info("segment_ids: {}".format(len(segment_ids)))
            logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                choices_features=choices_features,
                label=label
            )
        )
    return features

#获取特定的特征数据
def select_field(features, field):
    return [
        feature.choices_features[field] for feature in features
    ]

#定义模型
class NeuralNet(nn.Module):
    def __init__(self, model_name_or_path, hidden_size=768, num_class=2):
        super(NeuralNet, self).__init__()

        self.config = BertConfig.from_pretrained(model_name_or_path, num_labels=num_class)
        self.config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(model_name_or_path, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        # self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size*2, num_class)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.2) for _ in range(5)
        ])

    def forward(self, input_ids, input_mask, segment_ids, y=None, loss_fn=None):
        output = self.bert(input_ids, token_type_ids=segment_ids,
                                                                attention_mask=input_mask)
        last_hidden = output.last_hidden_state
        all_hidden_states = output.hidden_states
        batch_size = input_ids.shape[0]
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(
            13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(
            13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
        f = torch.mean(last_hidden, 1)
        feature = torch.cat((feature, f), 1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(feature))
                if loss_fn is not None:
                    loss = loss_fn(h, y)
            else:
                hi = self.fc(dropout(feature))
                h = h + hi
                if loss_fn is not None:
                    loss = loss + loss_fn(hi, y)
        if loss_fn is not None:
            return h / len(self.dropouts), loss / len(self.dropouts)
        return h / len(self.dropouts)

#RAdam优化
class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                            N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

#评价标准
def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1

#设置学习速率
def set_lr(optimizer, value):
    for p in optimizer.param_groups:
        p['lr'] = value
'''
https://fyubang.com/2019/06/01/ema/
用EMA（指数移动平均）这个方法对模型的参数做平均，以求提高测试指标并增加模型鲁棒。
EMA可以近似看成过去1/(1−α)个时刻v值的平均。
'''

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay#衰减
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

#问题后处理
def prob_postprocess(y_pred):
    prior = np.array([0.6903327690476333, 0.3096672309523667]) # 训练集 oppo正负样本比例
    y_pred_uncertainty = -(y_pred * np.log(y_pred)).sum(1) / np.log(2)

    threshold = 0.95
    y_pred_confident = y_pred[y_pred_uncertainty < threshold]
    y_pred_unconfident = y_pred[y_pred_uncertainty >= threshold]

    right, alpha, iters = 0, 1, 1
    post = []
    for i, y in enumerate(y_pred_unconfident):
        Y = np.concatenate([y_pred_confident, y[None]], axis=0)
        for j in range(iters):
            Y = Y ** alpha
            Y /= Y.sum(axis=0, keepdims=True)
            Y *= prior[None]
            Y /= Y.sum(axis=1, keepdims=True)
        y = Y[-1]
        post.append(y.tolist())

    post = np.array(post)
    y_pred[y_pred_uncertainty >= threshold] = post

    return y_pred



# 加载数据
tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
train_examples = read_examples(train_input, split='train')
train_features = convert_examples_to_features(
    train_examples, tokenizer, max_seq_length, True)

all_input_ids = np.array(select_field(train_features, 'input_ids'))
logger.info('shape: {}'.format(all_input_ids.shape))
all_input_mask = np.array(select_field(train_features, 'input_mask'))
all_segment_ids = np.array(select_field(train_features, 'segment_ids'))
all_label = np.array([f.label for f in train_features])
logger.info(Counter(all_label))


test_examples = read_examples(test_input, split='test')
test_features = convert_examples_to_features(
    test_examples, tokenizer, max_seq_length, True)
test_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
test_input_mask = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
test_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
#分层采样交叉切分，确保训练集，测试集中各类别样本的比例与原始数据集中相同
oof_train = np.zeros((len(train_examples), 2), dtype=np.float32)
oof_test = np.zeros((len(test_examples), 2), dtype=np.float32)


for fold, (train_index, valid_index) in enumerate(skf.split(all_label, all_label)):
    logger.info('================     fold {}        ==============='.format(fold))

    # 处理模型输入数据
    train_input_ids = torch.tensor(all_input_ids[train_index], dtype=torch.long)
    train_input_mask = torch.tensor(all_input_mask[train_index], dtype=torch.long)
    train_segment_ids = torch.tensor(all_segment_ids[train_index], dtype=torch.long)
    train_label = torch.tensor(all_label[train_index], dtype=torch.long)

    valid_input_ids = torch.tensor(all_input_ids[valid_index], dtype=torch.long)
    valid_input_mask = torch.tensor(all_input_mask[valid_index], dtype=torch.long)
    valid_segment_ids = torch.tensor(all_segment_ids[valid_index], dtype=torch.long)
    valid_label = torch.tensor(all_label[valid_index], dtype=torch.long)

    train = torch.utils.data.TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label)
    valid = torch.utils.data.TensorDataset(valid_input_ids, valid_input_mask, valid_segment_ids, valid_label)
    test = torch.utils.data.TensorDataset(test_input_ids, test_input_mask, test_segment_ids)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    model = NeuralNet(model_name_or_path).cuda()#创建选取的模型
    model.cuda()
    # model = nn.DataParallel(model, device_ids=[0, 1])
    loss_fn = torch.nn.CrossEntropyLoss()#交叉熵损失函数

    # 优化器定义
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = RAdam(optimizer_grouped_parameters, lr=learning_rate, eps=1e-6)#优化器
    total_steps = num_epochs * len(train_loader)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader)*2, num_training_steps=total_steps)

    scaler = GradScaler()#GradScaler用于动态图模式下的"自动混合精度"的训练。它控制loss的缩放比例，有助于避免浮点数溢出的问题。

    best_f1 = 0.
    valid_best = np.zeros((valid_label.size(0), 2))

    early_stop = 0
    ema = EMA(model, 0.999)#提高鲁棒性，权重移动平均
    ema.register()
    for epoch in range(num_epochs):#开始训练数据
        train_loss = 0.
        lr_list = []
        # if epoch > 2:
        #     set_lr(optimizer, 2e-5)
        model.train()
        for batch in tqdm(train_loader):#tqdm加入一个进度条
            optimizer.zero_grad()
            batch = tuple(t.cuda() for t in batch)
            x_ids, x_mask, x_sids, y_truth = batch
            with autocast():
                y_pred, loss = model(x_ids, x_mask, x_sids, y=y_truth, loss_fn=loss_fn)
            scaler.scale(loss.mean()).backward()
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            ema.update()
            # skip_lr_sched = (scale != scaler.get_scale())
            # if not skip_lr_sched:
            #     scheduler.step()
            train_loss += loss.mean().item() / len(train_loader)

        ema.apply_shadow()
        model.eval()#使用model.eval()切换到测试模式
        """
        如果模型中有BN层(Batch Normalization）和Dropout，
        需要在训练时添加model.train()，在测试时添加model.eval()。
        其中model.train()是保证BN层用每一批数据的均值和方差，
        而model.eval()是保证BN用全部训练数据的均值和方差；
        而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，
        而model.eval()是利用到了所有网络连接。
        """
        val_loss = 0.
        valid_preds_fold = np.zeros((valid_label.size(0), 2))
        with torch.no_grad():
            for i, batch in tqdm(enumerate(valid_loader)):
                batch = tuple(t.cuda() for t in batch)
                x_ids, x_mask, x_sids, y_truth = batch
                with autocast():
                    y_pred, loss = model(x_ids, x_mask, x_sids, y_truth, loss_fn)
                    y_pred = y_pred.detach()
                    val_loss += loss.mean().item() / len(valid_loader)
                valid_preds_fold[i * batch_size:(i + 1) * batch_size] = F.softmax(y_pred, dim=1).cpu().numpy()
        acc, f1 = metric(all_label[valid_index], np.argmax(valid_preds_fold, axis=1))
        if best_f1 < f1:
            early_stop = 0
            best_f1 = f1
            valid_best = valid_preds_fold
            torch.save(model.state_dict(), './model_save/ernie_' + file_name + '_{}.bin'.format(fold))
        else:
            early_stop += 1
        logger.info(
            'epoch: %d, train loss: %.8f, valid loss: %.8f, acc: %.8f, f1: %.8f, best_f1: %.8f\n' %
            (epoch, train_loss, val_loss, acc, f1, best_f1))
        torch.cuda.empty_cache()  # 每个epoch结束之后清空显存，防止显存不足

        # 检测早停
        if early_stop >= patience:
            break

    # 得到一折模型对测试集的预测结果
    model.load_state_dict(torch.load('./model_save/ernie_' + file_name + '_{}.bin'.format(fold)))
    test_preds_fold = np.zeros((len(test_examples), 2))
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            batch = tuple(t.cuda() for t in batch)
            x_ids, x_mask, x_sids = batch
            with autocast():
                y_pred = model(x_ids, x_mask, x_sids).detach()
            test_preds_fold[i * batch_size:(i + 1) * batch_size] = F.softmax(y_pred, dim=1).cpu().numpy()

    oof_train[valid_index] = valid_best
    acc, f1 = metric(all_label[valid_index], np.argmax(valid_best, axis=1))
    logger.info('epoch: best, acc: %.8f, f1: %.8f, best_f1: %.8f\n' %
                (acc, f1, best_f1))
    oof_test += test_preds_fold / 5


# 保存概率文件
np.savetxt('./submit/train_prob/train_bert_' + file_name + '.txt', oof_train)
np.savetxt('./submit/test_prob/test_bert_' + file_name + '.txt', oof_test)
acc, f1 = metric(all_label, np.argmax(oof_train, axis=1))
logger.info('epoch: best, acc: %.8f, f1: %.8f \n' % (acc, f1))

analysis = pd.DataFrame()
analysis['s1'] = [line.s1 for line in train_examples]
analysis['s2'] = [line.s2 for line in train_examples]
analysis['label'] = [line.label for line in train_examples]
analysis['pred'] = np.argmax(oof_train, axis=1).tolist()
analysis[analysis['label'] != analysis['pred']].to_csv('analysis_{}.csv'.format(f1), index=False)


# 后处理
oof_test = prob_postprocess(oof_test)
y_preds = np.argmax(oof_test, axis=1)
logger.info(Counter(y_preds))

with open('./output/predict_result_{}.csv'.format(f1), 'w', encoding="utf-8") as f:
    for y_pred in y_preds:
        f.write(str(y_pred) + "\n")


def compare_pinyin(s1, s2):
    s1_pinyin = ""
    s2_pinyin = ""
    for w in jieba.cut(s1):
        s1_pinyin += ''.join(lazy_pinyin(w))
    for w in jieba.cut(s2):
        s2_pinyin += ''.join(lazy_pinyin(w))
    return s1_pinyin == s2_pinyin


def postprocess(data, pred):
    post = []
    for line, lable in tqdm(zip(data, pred)):
        # r1 = correct(line.s1, line.s2)  # 339
        r2 = compare_pinyin(line.s1, line.s2)  # 339
        if r2:
            post.append(1)
        else:
            post.append(lable)
    post = np.array(post)
    print(np.count_nonzero(post != pred))
    return post

post = postprocess(test_examples, y_preds)

with open('./output/post_predict_result_{}.csv'.format(f1), 'w', encoding="utf-8") as f:
    for y_pred in post:
        f.write(str(y_pred) + "\n")