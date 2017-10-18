import time
import os
import numpy as np
import random

import logging
import torch
import torch.nn as nn
from torchtext import data

from args import get_args
from model import SmPlusPlus, PairwiseLossCriterion
from trec_dataset import TrecDataset
from evaluate import evaluate

args = get_args()
config = args

torch.manual_seed(args.seed)

def set_vectors(field, vector_path):
    if os.path.isfile(vector_path):
        stoi, vectors, dim = torch.load(vector_path)
        field.vocab.vectors = torch.Tensor(len(field.vocab), dim)

        for i, token in enumerate(field.vocab.itos):
            wv_index = stoi.get(token, None)
            if wv_index is not None:
                field.vocab.vectors[i] = vectors[wv_index]
            else:
                # initialize <unk> with U(-0.25, 0.25) vectors
                field.vocab.vectors[i] = torch.FloatTensor(dim).uniform_(-0.25, 0.25)
    else:
        print("Error: Need word embedding pt file")
        print("Error: Need word embedding pt file")
        exit(1)
    return field



# Set default configuration in : args.py
args = get_args()
config = args

# Set random seed for reproducibility
torch.manual_seed(args.seed)
if not args.cuda:
    args.gpu = -1
if torch.cuda.is_available() and args.cuda:
    print("Note: You are using GPU for training")
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cuda:
    print("You have Cuda but you're using CPU for training.")
np.random.seed(args.seed)
random.seed(args.seed)

QID = data.Field(sequential=False)
AID = data.Field(sequential=False)
QUESTION = data.Field(batch_first=True)
ANSWER = data.Field(batch_first=True)
LABEL = data.Field(sequential=False)
EXTERNAL = data.Field(sequential=False, tensor_type=torch.FloatTensor, batch_first=True, use_vocab=False,
                      preprocessing=data.Pipeline(lambda x: x.split()),
                      postprocessing=data.Pipeline(lambda x, train: [float(y) for y in x]))

train, dev, test = TrecDataset.splits(QID, QUESTION, AID, ANSWER, EXTERNAL, LABEL)

QID.build_vocab(train, dev, test)
AID.build_vocab(train, dev, test)
QUESTION.build_vocab(train, dev, test)
ANSWER.build_vocab(train, dev, test)
# POS.build_vocab(train, dev, test)
# NEG.build_vocab(train, dev, test)
LABEL.build_vocab(train, dev, test)


QUESTION = set_vectors(QUESTION, args.vector_cache)
ANSWER = set_vectors(ANSWER, args.vector_cache)

train_iter = data.Iterator(train, batch_size=args.batch_size, device=args.gpu, train=True, repeat=False,
                                   sort=False, shuffle=True)
dev_iter = data.Iterator(dev, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)
test_iter = data.Iterator(test, batch_size=args.batch_size, device=args.gpu, train=False, repeat=False,
                                   sort=False, shuffle=False)

config.target_class = len(LABEL.vocab)
config.questions_num = len(QUESTION.vocab)
config.answers_num = len(ANSWER.vocab)

print("Dataset {}    Mode {}".format(args.dataset, args.mode))
print("VOCAB num", len(QUESTION.vocab))
print("LABEL.target_class:", len(LABEL.vocab))
print("LABELS:", LABEL.vocab.itos)
print("Train instance", len(train))
print("Dev instance", len(dev))
print("Test instance", len(test))

if args.resume_snapshot:
    if args.cuda:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
else:
    model = SmPlusPlus(config)
    model.static_question_embed.weight.data.copy_(QUESTION.vocab.vectors)
    model.nonstatic_question_embed.weight.data.copy_(QUESTION.vocab.vectors)
    model.static_answer_embed.weight.data.copy_(ANSWER.vocab.vectors)
    model.nonstatic_answer_embed.weight.data.copy_(ANSWER.vocab.vectors)

    pw_model = PairwiseConv(model)

    if args.cuda:
        model.cuda()
        print("Shift model to GPU")

parameter = filter(lambda p: p.requires_grad, model.parameters())

# the SM model originally follows SGD but Adadelta is used here
optimizer = torch.optim.Adadelta(parameter, lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
pairwiseLoss = PairwiseLossCriterion()

early_stop = False
best_dev_map = 0
best_dev_loss = 0
iterations = 0
iters_not_improved = 0
epoch = 0
start = time.time()
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(os.path.join(args.save_path, args.dataset), exist_ok=True)
print(header)

index2label = np.array(LABEL.vocab.itos)
index2qid = np.array(QID.vocab.itos)
index2question = np.array(ANSWER.vocab.itos)

while True:
    if early_stop:
        print("Early Stopping. Epoch: {}, Best Dev Loss: {}".format(epoch, best_dev_loss))
        break
    epoch += 1
    train_iter.init_epoch()
    for batch_idx, batch in enumerate(train_iter):
        iterations += 1
        loss_num = 0
        # model.train();
        pw_model.train();
        optimizer.zero_grad()
        output = pw_model(batch)
        loss = pairwiseLoss(output)
        loss_num += loss.data[0]
        loss.backwward()
        # scores = model(batch)
        # n_correct += (torch.max(scores, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
        # n_total += batch.batch_size
        # train_acc = 100. * n_correct / n_total
        # loss = criterion(scores, batch.label)
        # loss.backward()
        optimizer.step()

        # Evaluate performance on validation set
        if iterations % args.dev_every == 1:
            # switch model into evaluation mode
            # model.eval()
            pw_model.eval()
            dev_iter.init_epoch()
            # n_dev_correct = 0
            dev_losses = []
            instance = []
            dev_loss_num = 0
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                # qid_array = index2qid[np.transpose(dev_batch.qid.cpu().data.numpy())]
                output = pw_model(batch)
                loss = pairwiseLoss(output)
                dev_loss_num += loss.data[0]
                loss.backwward()
                # true_label_array = index2label[np.transpose(dev_batch.label.cpu().data.numpy())]
                # scores = model(dev_batch)
                # n_dev_correct += (torch.max(scores, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                # dev_loss = criterion(scores, dev_batch.label)
                # dev_losses.append(dev_loss.data[0])
                # index_label = np.transpose(torch.max(scores, 1)[1].view(dev_batch.label.size()).cpu().data.numpy())
                # label_array = index2label[index_label]
                # get the relevance scores
                # score_array = scores[:, 2].cpu().data.numpy()
                # for i in range(dev_batch.batch_size):
                #     this_qid, predicted_label, score, gold_label = qid_array[i], label_array[i], score_array[i], true_label_array[i]
                #     instance.append((this_qid, predicted_label, score, gold_label))

            # dev_map, dev_mrr = evaluate(instance, 'valid', config.mode)

            # print(dev_log_template.format(time.time() - start,
            #                               epoch, iterations, 1 + batch_idx, len(train_iter),
            #                               100. * (1 + batch_idx) / len(train_iter), loss.data[0],
            #                               sum(dev_losses) / len(dev_losses), train_acc, dev_map))

            # Update validation results
            if dev_loss_num < best_dev_loss:
                iters_not_improved = 0
                best_dev_loss = dev_loss_num
                snapshot_path = os.path.join(args.save_path, args.dataset, args.mode+'_best_model.pt')
                torch.save(model, snapshot_path)
            else:
                iters_not_improved += 1
                if iters_not_improved >= args.patience:
                    early_stop = True
                    break

        if iterations % args.log_every == 1:
            # print progress message
            print(log_template.format(time.time() - start,
                                      epoch, iterations, 1 + batch_idx, len(train_iter),
                                      100. * (1 + batch_idx) / len(train_iter), loss_num, ' ' * 8,
                                      dev_loss_num, ' ' * 12))
            # print(log_template.format(time.time() - start,
            #                           epoch, iterations, 1 + batch_idx, len(train_iter),
            #                           100. * (1 + batch_idx) / len(train_iter), loss.data[0], ' ' * 8,
            #                           n_correct / n_total * 100, ' ' * 12))
