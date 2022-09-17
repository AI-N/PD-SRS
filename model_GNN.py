
import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tqdm import tqdm


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.norm = opt.norm
        self.ta = opt.TA
        self.scale = opt.scale
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        if self.ta:
            self.linear_t = nn.Linear(self.hidden_size, self.hidden_size, bias=False)  # target attention
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        if self.norm:
            # norms = torch.norm(a, p=2, dim=1, keepdim=True)  # a needs to be normalized too
            # a = a.div(norms)
            norms = torch.norm(self.embedding.weight, p=2, dim=1).data  # l2 norm over item embedding again for b
            self.embedding.weight.data = self.embedding.weight.data.div(norms.view(-1, 1).expand_as(self.embedding.weight))
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        if self.ta:
            qt = self.linear_t(hidden)  # batch_size x seq_length x latent_size
            beta = F.softmax(b @ qt.transpose(1, 2), -1)  # batch_size x n_nodes x seq_length
            target = beta @ hidden  # batch_size x n_nodes x latent_size
            a = a.view(ht.shape[0], 1, ht.shape[1])  # b,1,d
            a = a + target  # b,n,d
            scores = torch.sum(a * b, -1)  # b,n
        else:
            scores = torch.matmul(a, b.transpose(1, 0))
        if self.scale:
            scores = 16 * scores  # 16 is the sigma factor
        return scores

    def forward(self, inputs, A):
        if self.norm:
            norms = torch.norm(self.embedding.weight, p=2, dim=1).data  # l2 norm over item embedding
            self.embedding.weight.data = self.embedding.weight.data.div(norms.view(-1, 1).expand_as(self.embedding.weight))
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)
    # if model.norm:
    #     seq_shape = list(hidden.size())
    #     hidden = hidden.view(-1, model.hidden_size)
    #     norms = torch.norm(hidden, p=2, dim=1)  # l2 norm over session embedding
    #     hidden = hidden.div(norms.unsqueeze(-1).expand_as(hidden))
    #     hidden = hidden.view(seq_shape)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    if model.norm:
        seq_shape = list(seq_hidden.size())
        seq_hidden = seq_hidden.view(-1, model.hidden_size)
        norms = torch.norm(seq_hidden, p=2, dim=1)  # l2 norm over session embedding
        seq_hidden = seq_hidden.div(norms.unsqueeze(-1).expand_as(seq_hidden))
        seq_hidden = seq_hidden.view(seq_shape)
    return targets, model.compute_scores(seq_hidden, mask)



def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit5, mrr5, hit10, mrr10, hit20, mrr20 = [], [], [], [], [], []
    slices = test_data.generate_batch(model.batch_size)
    #scores5_ind=[]
    scores5_i , scores5_val, scores10_i, scores10_val, scores20_i, scores20_val, scores5000_i, scores5000_val = [], [], [], [], [], [], [], []
    target_=[]
    for i in slices:
        targets, scores = forward(model, i, test_data)
        scores5_val.append(trans_to_cpu(scores.topk(5)[0]).detach().numpy())
        scores10_val.append(trans_to_cpu(scores.topk(10)[0]).detach().numpy())
        scores20_val.append(trans_to_cpu(scores.topk(20)[0]).detach().numpy())
        scores5000_val.append(trans_to_cpu(scores.topk(5000)[0]).detach().numpy())
        sub_scores5 = trans_to_cpu(scores.topk(5)[1]).detach().numpy()
        sub_scores10 = trans_to_cpu(scores.topk(10)[1]).detach().numpy()
        sub_scores20 = trans_to_cpu(scores.topk(20)[1]).detach().numpy()
        sub_scores500 = trans_to_cpu(scores.topk(5000)[1]).detach().numpy()
        scores5_i.append(sub_scores5)
        scores10_i.append(sub_scores10)
        scores20_i.append(sub_scores20)
        scores5000_i.append(sub_scores500)

        target_.append(targets)

        #np.concatenate([scores5_ind,sub_scores5])
        for score, target, mask in zip(sub_scores5, targets, test_data.mask):
            hit5.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr5.append(0)
            else:
                mrr5.append(1 / (np.where(score == target - 1)[0][0] + 1))  ##position of item in the list: the beter the topper it sits
        for score, target, mask in zip(sub_scores10, targets, test_data.mask):
            hit10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr10.append(0)
            else:
                mrr10.append(1 / (np.where(score == target - 1)[0][0] + 1))
        for score, target, mask in zip(sub_scores20, targets, test_data.mask):
            hit20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr20.append(0)
            else:
                mrr20.append(1 / (np.where(score == target - 1)[0][0] + 1))

    scores5_value=[item for sublist in scores5_val for item in sublist]
    scores5_ind=[item for sublist in scores5_i for item in sublist]
    scores10_value=[item for sublist in scores10_val for item in sublist]
    scores10_ind=[item for sublist in scores10_i for item in sublist]
    scores20_value=[item for sublist in scores20_val for item in sublist]
    scores20_ind=[item for sublist in scores20_i for item in sublist]
    scores5000_value=[item for sublist in scores5000_val for item in sublist]
    scores5000_ind=[item for sublist in scores5000_i for item in sublist]
    targets_=[item for sublist in target_ for item in sublist]   

    hit5 = np.mean(hit5) * 100
    mrr5 = np.mean(mrr5) * 100
    hit10 = np.mean(hit10) * 100
    mrr10 = np.mean(mrr10) * 100
    hit20 = np.mean(hit20) * 100
    mrr20 = np.mean(mrr20) * 100
    return hit5, mrr5, hit10, mrr10, hit20, mrr20, targets_, scores5_value, scores5_ind, scores10_value, scores10_ind, scores20_value, scores20_ind, scores5000_value, scores5000_ind

