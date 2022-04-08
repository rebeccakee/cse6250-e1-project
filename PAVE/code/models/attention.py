import os
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import *
import torch.nn.init as init
import numpy as np
import sys
sys.path.append('tools')
sys.path.append('models')
import model_function
import parse, py_op






class ScaledDotProductAttentionClass(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttentionClass, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(2)
        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, q_data, k_data, v_data, attn_mask=None):
        attn = torch.bmm(q_data, k_data.transpose(1, 2)) / self.temper
        if attn_mask is not None:
            attn.data.masked_fill_(attn_mask, -float('inf'))
        attn, output = self.forward_func(v_data, attn)
        return output, attn

    def forward_func(self, v_data, attn):
        attn = self.softmax(attn)
        pattn = self.pooling(attn)
        pattn = pattn.expand(attn.size())
        mask = np.zeros(attn.size(), dtype=np.float32)
        mask[pattn.data.cpu().numpy() == attn.data.cpu().numpy()] = 1
        mask = Variable(torch.from_numpy(mask).cuda())
        attn = attn * mask
        attn = self.dropout(attn)
        output = torch.bmm(attn, v_data)
        return attn,output


class MultiHeadAttentionClass(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttentionClass, self).__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))

        self.attention = ScaledDotProductAttentionClass(d_model)
        self.proj = nn.Linear(n_head*d_v, d_model)

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model * 2, d_model )

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, q_data, k_data, v_data, attn_mask=None):

        temp_d_k, temp_d_v = self.d_k, self.d_v
        temp_n_head = self.n_head
        residual_data = q_data
        outputs, attns = self.forward_func(q_data, k_data, v_data, temp_d_k, temp_d_v, temp_n_head, residual_data)
        return outputs, attns.data.cpu().numpy()

    def forward_func(self, q_data, k_data, v_data, temp_d_k, temp_d_v, temp_n_head, residual_data):
        mb_size, len_q, d_model = q_data.size()
        mb_size, len_k, d_model = k_data.size()
        mb_size, len_v, d_model = v_data.size()
        q_s = q_data.repeat(temp_n_head, 1, 1).view(temp_n_head, -1, d_model) 
        k_s = k_data.repeat(temp_n_head, 1, 1).view(temp_n_head, -1, d_model) 
        v_s = v_data.repeat(temp_n_head, 1, 1).view(temp_n_head, -1, d_model) 
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, temp_d_k)   
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, temp_d_k)   
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, temp_d_v)   
        outputs, attns = self.attention(q_s, k_s, v_s)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1) 
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)
        outputs = self.linear(torch.cat((residual_data, outputs), 2))
        return outputs,attns
        


def value_embedding_data(d = 512, split = 100):
    vec = np.array([np.arange(split) * i for i in range(int(d/2))], dtype=np.float32).transpose()
    vec = vec / vec.max() 
    embedding = np.concatenate((np.sin(vec), np.cos(vec)), 1)
    embedding[0, :d] = 0
    embedding = torch.from_numpy(embedding)
    return embedding


class MyAttention(nn.Module):
    def __init__(self, arg_list):
        super ( MyAttention, self ).__init__ ( )
        self.args = arg_list
        self.vital_embedding = nn.Embedding (arg_list.vocab_size, arg_list.embed_size ) 

        self.value_embedding = nn.Embedding.from_pretrained(value_embedding_data(arg_list.embed_size, arg_list.n_split + 1))
        self.xv_mapping = nn.Sequential( nn.Linear (2 * arg_list.embed_size, arg_list.embed_size ) , nn.ReLU(), nn.Linear (arg_list.embed_size, arg_list.embed_size ) )
        self.q_mapping = nn.Sequential( nn.Linear (arg_list.embed_size, arg_list.embed_size ) , nn.ReLU(), nn.Linear (arg_list.embed_size, arg_list.embed_size ) )
        self.k_mapping = nn.Sequential( nn.Linear (arg_list.embed_size, arg_list.embed_size ) , nn.ReLU(), nn.Linear (arg_list.embed_size, arg_list.embed_size ) )
        self.v_mapping = nn.Sequential( nn.Linear (arg_list.embed_size, arg_list.embed_size ) , nn.ReLU(), nn.Linear (arg_list.embed_size, arg_list.embed_size ) )
        self.attention_mapping = nn.Sequential( nn.Linear (arg_list.embed_size, arg_list.embed_size ) , nn.ReLU(), nn.Linear (arg_list.embed_size, arg_list.embed_size ) )
        self.paw = nn.Sequential( nn.Linear (arg_list.embed_size, arg_list.embed_size ) , nn.ReLU(), nn.Linear (arg_list.embed_size, 1), nn.Softmax(1))
        self.sigmoid = nn.Sigmoid()
        self.master_embedding= nn.Linear(arg_list.master_size, arg_list.embed_size ) 
        self.time_encoding = nn.Sequential (
            nn.Embedding.from_pretrained(model_function.time_encoding_data(arg_list.embed_size, arg_list.time_range)),
            nn.Linear ( arg_list.embed_size, arg_list.embed_size),
            nn.ReLU ( ),
            nn.Dropout ( 0.25 ),
            nn.Linear ( arg_list.embed_size, arg_list.embed_size)
            )
        self.relu = nn.ReLU ( )

        if arg_list.use_glp:
            self.linears = nn.Sequential (
                nn.Linear ( arg_list.hidden_size * 2, arg_list.rnn_size ),
                nn.ReLU ( ),
                nn.Dropout ( 0.25 ),
                nn.Linear ( arg_list.rnn_size, 1),
            )
        else:
            self.linears = nn.Linear(arg_list.hidden_size * 2, 1, bias=False)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.pooling_with_indices = nn.AdaptiveMaxPool1d(1, return_indices = True)
        self.att_list = [MultiHeadAttentionClass(1, arg_list.embed_size, 128, 128)  for _ in range(arg_list.num_layers)]
        if arg_list.gpu:
            self.att_list = [x.cuda() for x in self.att_list]
        self.linear = nn.Linear ( arg_list.embed_size, 1, bias=False)
        self.linear_time = nn.Sequential(
				nn.Linear ( arg_list.embed_size, arg_list.embed_size),
                nn.ReLU ( ),
				nn.Linear ( arg_list.embed_size, arg_list.embed_size),
                nn.Dropout ( 0.1 ),
                nn.ReLU ( )
				)

    def visit_pooling(self, output):
        size = output.size()
        output = output.view(size[0] * size[1], size[2], output.size(3)) 
        output = torch.transpose(output, 1,2).contiguous() 
        output = self.pooling(output)
        output = output.view(size[0], size[1], size[3])
        return output

    def max_pooling_with_dim(self, x, dim):
        size = list(x.size())
        s1, s3 = 1, 1
        for d in range(dim):
            s1 *= size[d]
        for d in range(dim + 1, len(size)):
            s3 *= size[d]
        x = x.view(s1, size[dim], s3)
        x = torch.transpose(x, 1,2).contiguous()
        x, idx = self.pooling_with_indices(x)
        
        new_size = size[:dim] + size[dim+1:]
        x = x.view(new_size)
        return x, idx


    def attend_demo(self, xv, m, x):
        if self.args.use_value_embedding:
            x = xv[:, :, 0, :].contiguous()
            v = xv[:, :, 1, :].contiguous()
            x_size = list(x.size())                         
            x = x.view((x_size[0], -1))                     
            v = v.view((x_size[0], -1))                     
            x = self.vital_embedding(x)                     
            v = self.value_embedding(v)
            x = self.xv_mapping(torch.cat((x,v), 2)).contiguous()
        else:
            x_size = list(x.size())                         
            x = x.view((x_size[0], -1))                     
            x = self.vital_embedding(x)

        m = self.master_embedding(m)                    
        m = m.view((m.size(0), 1, m.size(1)))                  
        m = m.expand(x.size())                            
        k = self.attention_mapping(x)                   
        a = self.sigmoid(k * m)                         
        x = x + a * m                                   
        
        return x

    def attend_pattern(self, x):
        paw = self.paw(x)                               
        paw = paw.transpose(1,2)                        
        x = torch.bmm(paw, x)                           
        x = x.view((x.size(0), -1))
        return x, paw




    def forward(self, x, master, mask=None, time=None, phase='train', value=None, trend=None):
        args = self.args
        x_size = list(x.size())                        
        a_x = self.attend_demo(value, master, x)               
        q_x = self.q_mapping(a_x)                       
        k_x = self.k_mapping(a_x)                       
        v_x = self.v_mapping(a_x)                       
        time = - time.long()
        e_t = self.time_encoding(time)                
        e_t = e_t.unsqueeze(2).contiguous()
        e_t = e_t.expand(x_size + [e_t.size(3)]).contiguous()
        e_t = e_t.view(x_size[0], -1, args.embed_size)    
        q_x = q_x.view(x_size[0], -1, args.embed_size)    
        k_x = k_x.view(x_size[0], -1, args.embed_size)    
        v_x = v_x.view(x_size[0], -1, args.embed_size)    
        q_x = q_x + e_t
        attn_list = []
        for i_a, att in enumerate(self.att_list):
            k_x = k_x + e_t
            k_x, attn = att(q_x, k_x, v_x)
            attn_list.append(attn)

        if args.use_gp:
            mout, idx = self.max_pooling_with_dim(k_x, 1)
            out = self.linear(mout)
            return [out]
        else:
            out, paw = self.attend_pattern(k_x)
            out = self.linear(out)
            return [out]
