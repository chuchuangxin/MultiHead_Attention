import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, X):
        batch_size, seq_length, _ = X.shape

        Q = self.Wq(X).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(X).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(X).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(
            torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool(), float('-inf')
        )

        attention_weights = torch.softmax(attention_scores, dim=-1)

        A = torch.matmul(attention_weights, V)
        A = A.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.Wo(A)

        return output, attention_weights


# 测试代码
def test_multihead_attention():
    batch_size = 4
    seq_length = 16
    d_model = 64
    num_heads = 4

    mha = MultiHeadSelfAttention(d_model, num_heads)
    X = torch.rand(batch_size, seq_length, d_model)  # 生成随机输入
    output, attn_weights = mha(X)

    print("Output shape:", output.shape)  # 期望: (4, 16, 64)
    print("Attention weights shape:", attn_weights.shape)  # 期望: (4, 4, 16, 16)
    print("Output tensor:", output)
    print("Attention weights tensor:", attn_weights)

# 运行测试
test_multihead_attention()
