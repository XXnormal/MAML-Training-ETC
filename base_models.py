import torch.nn as nn
import torch
import math

class EmbbedingNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=32, output_dim=32):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.bn_1 = nn.BatchNorm1d(hidden_dim)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        self.bn_2 = nn.BatchNorm1d(output_dim)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.linear_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, vocal_size, input_dim=32, hidden_dim=128, num_classes=95, num_layers=6, dropout=0.1, max_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocal_size, input_dim)
        self.positional_encoding = PositionalEncoding(input_dim, max_len)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(input_dim, 8, hidden_dim, dropout, batch_first=True), 
            num_layers=num_layers,
        )
        self.linear = nn.Linear(input_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.feat_dim = input_dim

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.linear(x[:, 0, :])
        x = self.softmax(x)
        return x
    
    def featureExtract(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        return x[:, 0, :]

    def classify(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x

class DFNet(nn.Module):
    def __init__(self, length, classes):
        super(DFNet, self).__init__()

        self.length = length

        filter_num = [None, 32, 64, 128, 256]
        kernel_size = [None, 8, 8, 8, 8]
        conv_stride_size = [None, 1, 1, 1, 1]
        pool_stride_size = [None, 4, 4, 4, 4]
        pool_size = [None, 8, 8, 8, 8]

        # Block 1
        self.block1_conv1 = nn.Conv1d(in_channels=1, out_channels=filter_num[1], kernel_size=kernel_size[1], stride=conv_stride_size[1], padding=1)
        self.block1_bn1 = nn.BatchNorm1d(filter_num[1])
        self.block1_elu1 = nn.ELU(alpha=1.0)
        self.block1_conv2 = nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[1], kernel_size=kernel_size[1], stride=conv_stride_size[1], padding=1)
        self.block1_bn2 = nn.BatchNorm1d(filter_num[1])
        self.block1_elu2 = nn.ELU(alpha=1.0)
        self.block1_pool = nn.MaxPool1d(kernel_size=pool_size[1], stride=pool_stride_size[1], padding=1)
        self.block1_dropout = nn.Dropout(0.1)

        # Block 2
        self.block2_conv1 = nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[2], kernel_size=kernel_size[2], stride=conv_stride_size[2], padding=1)
        self.block2_bn1 = nn.BatchNorm1d(filter_num[2])
        self.block2_act1 = nn.ReLU()
        self.block2_conv2 = nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[2], kernel_size=kernel_size[2], stride=conv_stride_size[2], padding=1)
        self.block2_bn2 = nn.BatchNorm1d(filter_num[2])
        self.block2_act2 = nn.ReLU()
        self.block2_pool = nn.MaxPool1d(kernel_size=pool_size[2], stride=pool_stride_size[2], padding=1)
        self.block2_dropout = nn.Dropout(0.1)

        # Block 3
        self.block3_conv1 = nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[3], kernel_size=kernel_size[3], stride=conv_stride_size[3], padding=1)
        self.block3_bn1 = nn.BatchNorm1d(filter_num[3])
        self.block3_act1 = nn.ReLU()
        self.block3_conv2 = nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[3], kernel_size=kernel_size[3], stride=conv_stride_size[3], padding=1)
        self.block3_bn2 = nn.BatchNorm1d(filter_num[3])
        self.block3_act2 = nn.ReLU()
        self.block3_pool = nn.MaxPool1d(kernel_size=pool_size[3], stride=pool_stride_size[3], padding=1)
        self.block3_dropout = nn.Dropout(0.1)

        # Block 4
        self.block4_conv1 = nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[4], kernel_size=kernel_size[4], stride=conv_stride_size[4], padding=1)
        self.block4_bn1 = nn.BatchNorm1d(filter_num[4])
        self.block4_act1 = nn.ReLU()
        self.block4_conv2 = nn.Conv1d(in_channels=filter_num[4], out_channels=filter_num[4], kernel_size=kernel_size[4], stride=conv_stride_size[4], padding=1)
        self.block4_bn2 = nn.BatchNorm1d(filter_num[4])
        self.block4_act2 = nn.ReLU()
        self.block4_pool = nn.MaxPool1d(kernel_size=pool_size[4], stride=pool_stride_size[4], padding=1)
        self.block4_dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256 * 15, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc1_act = nn.ReLU()
        self.fc1_dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc2_act = nn.ReLU()
        self.fc2_dropout = nn.Dropout(0.1)

        self.fc3 = nn.Linear(128, classes)
        self.softmax = nn.Softmax(dim=1)

        self.feat_dim = 128

    def featureExtract(self, x):
        x = x.reshape(-1, 1, self.length)
        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = self.block1_elu1(x)
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        x = self.block1_elu2(x)
        x = self.block1_pool(x)
        x = self.block1_dropout(x)

        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = self.block2_act1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        x = self.block2_act2(x)
        x = self.block2_pool(x)
        x = self.block2_dropout(x)

        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = self.block3_act1(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_act2(x)
        x = self.block3_pool(x)
        x = self.block3_dropout(x)

        x = self.block4_conv1(x)
        x = self.block4_bn1(x)
        x = self.block4_act1(x)
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = self.block4_act2(x)
        x = self.block4_pool(x)
        x = self.block4_dropout(x)
        x = self.flatten(x)

        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.fc1_bn(x)
        x = self.fc1_act(x)
        x = self.fc1_dropout(x)

        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.fc2_bn(x)
        x = self.fc2_act(x)
        x = self.fc2_dropout(x)
        return x

    def classify(self, x):
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def forward(self, x):
        feat = self.featureExtract(x)
        out = self.classify(feat)
        return out

class DFTransformer(nn.Module):
    def __init__(self, length, classes):
        super(DFTransformer, self).__init__()

        self.length = length

        filter_num = [None, 32, 64, 128, 256]
        kernel_size = [None, 8, 8, 8, 8]
        conv_stride_size = [None, 1, 1, 1, 1]
        pool_stride_size = [None, 4, 4, 4, 4]
        pool_size = [None, 8, 8, 8, 8]

        # Block 1
        self.block1_conv1 = nn.Conv1d(in_channels=1, out_channels=filter_num[1], kernel_size=kernel_size[1], stride=conv_stride_size[1], padding=1)
        self.block1_bn1 = nn.BatchNorm1d(filter_num[1])
        self.block1_elu1 = nn.ELU(alpha=1.0)
        self.block1_conv2 = nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[1], kernel_size=kernel_size[1], stride=conv_stride_size[1], padding=1)
        self.block1_bn2 = nn.BatchNorm1d(filter_num[1])
        self.block1_elu2 = nn.ELU(alpha=1.0)
        self.block1_pool = nn.MaxPool1d(kernel_size=pool_size[1], stride=pool_stride_size[1], padding=1)
        self.block1_dropout = nn.Dropout(0.1)

        # Block 2
        self.block2_conv1 = nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[2], kernel_size=kernel_size[2], stride=conv_stride_size[2], padding=1)
        self.block2_bn1 = nn.BatchNorm1d(filter_num[2])
        self.block2_act1 = nn.ReLU()
        self.block2_conv2 = nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[2], kernel_size=kernel_size[2], stride=conv_stride_size[2], padding=1)
        self.block2_bn2 = nn.BatchNorm1d(filter_num[2])
        self.block2_act2 = nn.ReLU()
        self.block2_pool = nn.MaxPool1d(kernel_size=pool_size[2], stride=pool_stride_size[2], padding=1)
        self.block2_dropout = nn.Dropout(0.1)

        # Block 3
        self.block3_conv1 = nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[3], kernel_size=kernel_size[3], stride=conv_stride_size[3], padding=1)
        self.block3_bn1 = nn.BatchNorm1d(filter_num[3])
        self.block3_act1 = nn.ReLU()
        self.block3_conv2 = nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[3], kernel_size=kernel_size[3], stride=conv_stride_size[3], padding=1)
        self.block3_bn2 = nn.BatchNorm1d(filter_num[3])
        self.block3_act2 = nn.ReLU()
        self.block3_pool = nn.MaxPool1d(kernel_size=pool_size[3], stride=pool_stride_size[3], padding=1)
        self.block3_dropout = nn.Dropout(0.1)

        # Block 4
        self.block4_conv1 = nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[4], kernel_size=kernel_size[4], stride=conv_stride_size[4], padding=1)
        self.block4_bn1 = nn.BatchNorm1d(filter_num[4])
        self.block4_act1 = nn.ReLU()
        self.block4_conv2 = nn.Conv1d(in_channels=filter_num[4], out_channels=filter_num[4], kernel_size=kernel_size[4], stride=conv_stride_size[4], padding=1)
        self.block4_bn2 = nn.BatchNorm1d(filter_num[4])
        self.block4_act2 = nn.ReLU()
        self.block4_pool = nn.MaxPool1d(kernel_size=pool_size[4], stride=pool_stride_size[4], padding=1)
        self.block4_dropout = nn.Dropout(0.1)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(filter_num[-1], 8, 128, 0.5, batch_first=True), 
            num_layers=6,
        )
        self.linear = nn.Linear(filter_num[-1], classes)
        self.softmax = nn.Softmax(dim=1)

        self.feat_dim = filter_num[-1]

    def featureExtract(self, x):
        x = x.reshape(-1, 1, self.length)
        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = self.block1_elu1(x)
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        x = self.block1_elu2(x)
        x = self.block1_pool(x)
        x = self.block1_dropout(x)

        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = self.block2_act1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        x = self.block2_act2(x)
        x = self.block2_pool(x)
        x = self.block2_dropout(x)

        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = self.block3_act1(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_act2(x)
        x = self.block3_pool(x)
        x = self.block3_dropout(x)

        x = self.block4_conv1(x)
        x = self.block4_bn1(x)
        x = self.block4_act1(x)
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = self.block4_act2(x)
        x = self.block4_pool(x)
        x = self.block4_dropout(x)
        
        x = x.transpose(1, 2)
        x = self.encoder(x)

        return x[:, 0, :]

    def forward(self, x):
        feat = self.featureExtract(x)
        out = self.linear(feat)
        out = self.softmax(out)
        return out
    
    def classify(self, x):
        x = self.linear(x)
        x = self.softmax(x)
        return x
    
class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size=256, projection_size=64):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)
