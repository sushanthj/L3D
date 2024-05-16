import torch
import torch.nn as nn
import torch.nn.functional as F

## OG ##

################# DGCNN Model ################################

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

################# DGCNN Refactored ################################

def nearest_neighbors(input_tensor, num_neighbors):
    inner_product = -2*torch.matmul(input_tensor.transpose(2, 1), input_tensor)
    squared_norm = torch.sum(input_tensor**2, dim=1, keepdim=True)
    distance_matrix = -squared_norm - inner_product - squared_norm.transpose(2, 1)

    indices = distance_matrix.topk(k=num_neighbors, dim=-1)[1]
    return indices


def compute_graph_features(input_tensor, num_neighbors=20, indices=None):
    batch_size = input_tensor.size(0)
    num_points = input_tensor.size(2)
    input_tensor = input_tensor.view(batch_size, -1, num_points)
    if indices is None:
        indices = nearest_neighbors(input_tensor, k=num_neighbors)
    device = torch.device('cuda')

    base_indices = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    indices = indices + base_indices

    indices = indices.view(-1)

    _, num_dims, _ = input_tensor.size()

    input_tensor = input_tensor.transpose(2, 1).contiguous()
    features = input_tensor.view(batch_size*num_points, -1)[indices, :]
    features = features.view(batch_size, num_points, num_neighbors, num_dims)
    input_tensor = input_tensor.view(batch_size, num_points, 1, num_dims).repeat(1, 1, num_neighbors, 1)

    features = torch.cat((features-input_tensor, input_tensor), dim=3).permute(0, 3, 1, 2).contiguous()

    return features


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        x = compute_graph_features(x)
        return self.conv(x)

class DGCNN(nn.Module):
    def __init__(self, args, num_output_channels=40):
        super(DGCNN, self).__init__()
        self.num_neighbors = args.k
        self.convs = nn.ModuleList([
            GraphConv(6, 64),
            GraphConv(64*2, 64),
            GraphConv(64*2, 128),
            GraphConv(128*2, 256)
        ])
        self.conv1d = nn.Sequential(
            nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.linears = nn.Sequential(
            nn.Linear(args.emb_dims*2, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=args.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=args.dropout),
            nn.Linear(256, num_output_channels)
        )

    def forward(self, x):
        batch_size = x.size(0)
        features = []
        for conv in self.convs:
            x = conv(x)
            x_max = x.max(dim=-1, keepdim=False)[0]
            features.append(x_max)
        x = torch.cat(features, dim=1)
        x = self.conv1d(x)
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x_max, x_avg), 1)
        return self.linears(x)

########### Zoe Refactored ####################

def knn_graph(input_tensor, num_neighbors=10):
    batch_size, num_dims, num_points = input_tensor.shape

    distance_matrix = torch.cdist(input_tensor, input_tensor)
    _, indices = torch.topk(distance_matrix, k=num_neighbors+1, dim=1, largest=False)
    indices = indices[:, 1:]

    indices += torch.arange(0, batch_size, device="cuda").view(-1, 1, 1)*num_points
    indices = indices.view(-1)

    input_tensor = input_tensor.transpose(2, 1).contiguous()
    features = input_tensor.view(batch_size*num_points, -1)[:, indices]
    features = features.view(batch_size, num_points, num_neighbors, num_dims) 
    input_tensor = input_tensor.unsqueeze(2).repeat(1, 1, num_neighbors, 1)

    features = torch.cat((features-input_tensor, input_tensor), dim=-1).permute(0, 3, 1, 2).contiguous()

    return features

class ClassificationModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ClassificationModel, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(6, 64, 1, bias=False),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.Conv2d(64, 128, 1, bias=False),
            nn.Conv2d(128, 256, 1, bias=False),
            nn.Conv2d(512, 1024, 1, bias=False)
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(1024)
        ])

        self.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        outputs = []
        for conv, bn in zip(self.convs, self.bns):
            points = knn_graph(points)
            points = F.leaky_relu(bn(conv(points)), negative_slope=0.2)
            points = torch.amax(points, dim=-1, keepdim=False)
            outputs.append(points)

        out = torch.cat(outputs, dim=1)
        out = torch.amax(F.leaky_relu(self.bns-1), negative_slope=0.2, dim=-1, keepdim=False)
        out = self.fc(out)

        return out


class SegmentationModel(nn.Module):
    def __init__(self, num_classes=6):
        super(SegmentationModel, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(6, 64, 1, bias=False),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.Conv2d(64*2, 64, 1, bias=False),
            nn.Conv2d(64, 64, 1, bias=False),
            nn.Conv2d(64*2, 64, 1, bias=False),
            nn.Conv1d(192, 1024, 1, bias=False),
            nn.Conv1d(16, 64, 1, bias=False)
        ])

        self.batch_norm_layers = nn.ModuleList([
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm1d(1024),
            nn.BatchNorm1d(64)
        ])

        self.point_layer = nn.Sequential(
            nn.Conv1d(1280, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Conv1d(256, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, num_classes, 1, bias=False),
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_classes)
        '''
        outputs = []
        for conv, bn in zip(self.conv_layers, self.batch_norm_layers):
            points = knn_graph(points)
            points = F.leaky_relu(bn(conv(points)), negative_slope=0.2)
            points = torch.amax(points, dim=-1, keepdim=False)
            outputs.append(points)

        combined = torch.cat(outputs, dim=1)
        combined = knn_graph(combined)
        combined = torch.amax(F.leaky_relu(self.batch_norm_layers-1), dim=-1, keepdim=False)

        cat_vec = F.leaky_relu(self.batch_norm_layers-1, negative_slope=0.2)
        combined = torch.cat((combined, cat_vec), dim=1).repeat(1, 1, points.shape[1])

        out = torch.cat((outputs[0], outputs[1], outputs[2], combined), dim=1)
        out = self.point_layer(out)

        return out