from collections import OrderedDict
import torch
import torch.nn.init as init
from torch import nn
from torch.autograd import Function


class Embedding_MDL(nn.Module):

    def __init__(self, opt):
        super(Embedding_MDL, self).__init__()
        # parameters
        self.model = opt["model_name"]
        self.total_node_num = opt["total_node_num"]
        self.train_node_num = opt["train_node_num"]
        self.parameter_num = opt["parameter_num"]
        assert opt["data_vectors"].shape[0] == opt["total_node_num"]
        assert opt["hidden_layer_num"] == 1

        # save data vectors as an nn.Embedding format
        self.data_vectors = nn.Embedding(
            opt["data_vectors"].shape[0],
            opt["data_vectors"].shape[1]
        )
        self.data_vectors.weight.data = torch.from_numpy(
            opt["data_vectors"]
        ).float()
        self.data_vectors.weight.requires_grad = False

        self.U_NN = self.build_NN(
            opt["data_vectors"],
            opt["hidden_layer_num"],
            opt["hidden_size"]
        )
        # no bias for MDL-WIPS
        self.U = nn.Linear(
            opt["hidden_size"],
            opt["parameter_num"],
            bias=False
        )

    def initialization(self):

        for name, param in self.named_parameters():
            if 'data_vectors' in name:
                continue
            if 'ips_weight' in name:
                # init.uniform_(param, 0.0, 1.0/self.parameter_num)
                # **For simplicity**, the results of the paper came from this line.

                # Recommended (since it shows better results for most cases).
                init.uniform_(param, -0.5 / self.parameter_num,
                              0.5 / self.parameter_num)
                continue
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                # init.kaiming_uniform_(
                #     param, mode='fan_in', nonlinearity='relu')
                # init.uniform_(
                #     param, a=-1, b=1)
                init.xavier_uniform_(param, gain=4)
            else:
                raise Exception(name)

    def build_NN(self, given_data_vectors, hidden_layer_num, hidden_size):
        # NN = [("fc0", nn.Linear(given_data_vectors.shape[1], hidden_size))]
        NN = [("fc0", nn.Linear(given_data_vectors.shape[1], hidden_size, bias=False))]
        # for i in range(1, hidden_layer_num):
        #     NN.extend([(f"relu{i-1}", nn.ReLU(True)), (f"fc{i}", nn.Linear(hidden_size, hidden_size))])
        # NN.append((f"relu{hidden_layer_num-1}", nn.ReLU(True)))
        NN.append((f"tanh{hidden_layer_num-1}", nn.Tanh()))
        # NN.append((f"tanh{hidden_layer_num-1}", nn.ReLU()))

        # print(NN)
        return nn.Sequential(OrderedDict(NN))

    def forward(self, inputs):
        inputs = self.U_NN(self.data_vectors(inputs))
        e = self.U(inputs)
        # o: the coordinates of neighbor nodes.
        o = e.narrow(1, 1, e.size(1) - 1)
        # s: the coordinates of the center node. They are expanded to have the
        # same shape as o.
        s = e.narrow(1, 0, 1).expand_as(o)
        dists = self.distfn([s, o]).squeeze(-1)
        return -dists

    def embed(self):
        embeddings = self.U(self.U_NN(self.data_vectors.state_dict()[
                            'weight'])).data.cpu().numpy()
        return [embeddings]

    def get_similarity(self, inputs):
        return self.forward(inputs)


class MDL_WIPS(Embedding_MDL):
    """ Weighted Inner Product Similarity (WIPS)"""

    def __init__(self, opt):
        super(MDL_WIPS, self).__init__(opt)
        self.ips_weight = nn.Parameter(torch.zeros(opt["parameter_num"]))
        self.initialization()

    def distfn(self, input, w=None):
        u, v = input
        if w is None:
            w = self.ips_weight
        return -torch.sum(u * v * w, dim=-1)

    def get_ips_weight(self):
        return self.ips_weight.data.cpu().numpy()


class Embedding(nn.Module):

    def __init__(self, opt):
        super(Embedding, self).__init__()
        # parameters
        self.model = opt["model_name"]
        self.total_node_num = opt["total_node_num"]
        self.train_node_num = opt["train_node_num"]
        self.parameter_num = opt["parameter_num"]
        assert opt["data_vectors"].shape[0] == opt["total_node_num"]
        assert opt["hidden_layer_num"] >= 1

        # save data vectors as an nn.Embedding format
        self.data_vectors = nn.Embedding(
            opt["data_vectors"].shape[0], opt["data_vectors"].shape[1]
        )
        self.data_vectors.weight.data = torch.from_numpy(
            opt["data_vectors"]).float()
        self.data_vectors.weight.requires_grad = False

        self.U_NN = self.build_NN(
            opt["data_vectors"],
            opt["hidden_layer_num"],
            opt["hidden_size"]
        )
        self.U = nn.Linear(
            opt["hidden_size"],
            opt["parameter_num"],
            bias=False
        )

    def initialization(self):

        for name, param in self.named_parameters():
            if 'data_vectors' in name:
                continue
            if 'ips_weight' in name:
                # init.uniform_(param, 0.0, 1.0/self.parameter_num) # **For
                # simplicity**, the results of the paper came from this line.

                # Recommended (since it shows better results for most cases).
                init.uniform_(param, -0.5 / self.parameter_num,
                              0.5 / self.parameter_num)
                continue
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_uniform_(
                    param, mode='fan_in', nonlinearity='relu')
            else:
                raise Exception(name)

    def build_NN(self, given_data_vectors, hidden_layer_num, hidden_size):
        NN = [("fc0", nn.Linear(given_data_vectors.shape[1], hidden_size, bias=False))]
        for i in range(1, hidden_layer_num):
            NN.extend([(f"tanh{i-1}", nn.Tanh(True)), (f"fc{i}", nn.Linear(hidden_size, hidden_size))])
        NN.append((f"tanh{hidden_layer_num-1}", nn.Tanh()))
        # print(NN)
        return nn.Sequential(OrderedDict(NN))

    def forward(self, inputs):
        inputs = self.U_NN(self.data_vectors(inputs))
        e = self.U(inputs)
        # o: the coordinates of neighbor nodes.
        o = e.narrow(1, 1, e.size(1) - 1)
        # s: the coordinates of the center node. They are expanded to have the
        # same shape as o.
        s = e.narrow(1, 0, 1).expand_as(o)
        dists = self.distfn([s, o]).squeeze(-1)
        return -dists

    def embed(self):
        embeddings = self.U(self.U_NN(self.data_vectors.state_dict()[
                            'weight'])).data.cpu().numpy()
        return [embeddings]

    def get_similarity(self, inputs):
        return self.forward(inputs)


class WIPS(Embedding):
    """ Weighted Inner Product Similarity (WIPS)"""

    def __init__(self, opt):
        super(WIPS, self).__init__(opt)
        self.ips_weight = nn.Parameter(torch.zeros(opt["parameter_num"]))
        self.initialization()

    def distfn(self, input, w=None):
        u, v = input
        if w is None:
            w = self.ips_weight
        return -torch.sum(u * v * w, dim=-1)

    def get_ips_weight(self):
        return self.ips_weight.data.cpu().numpy()


class IPS(Embedding):
    """ Inner Product Similarity (IPS)"""

    def __init__(self, opt):
        super(IPS, self).__init__(opt)
        self.initialization()

    def distfn(self, input):
        u, v = input
        return -(torch.sum(u * v, dim=-1))


class SIPS(Embedding):
    """ Shifted Inner Product Similarity (SIPS)"""

    def __init__(self, opt):
        super(SIPS, self).__init__(opt)
        self.U = nn.Embedding(
            opt["train_node_num"],
            opt["parameter_num"] - 1,
        )
        self.U_bias = nn.Embedding(
            opt["train_node_num"],
            1
        )
        self.U = nn.Linear(opt["hidden_size"], opt["parameter_num"] - 1)
        self.U_bias = nn.Linear(opt["hidden_size"], 1)
        self.initialization()

    def distfn(self, input):
        u, u_bias, v, v_bias = input
        return -(torch.sum(u * v, dim=-1) + u_bias.squeeze(-1) + v_bias.squeeze(-1))

    def forward(self, inputs):
        inputs = self.U_NN(self.data_vectors(inputs))
        e = self.U(inputs)
        eb = self.U_bias(inputs)
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        ob = eb.narrow(1, 1, e.size(1) - 1)
        sb = eb.narrow(1, 0, 1).expand_as(ob)
        dists = self.distfn([s, sb, o, ob]).squeeze(-1)
        return -dists

    def embed(self):
        tmp = self.U_NN(self.data_vectors.state_dict()['weight'])
        return [self.U(tmp).data.cpu().numpy(), self.U_bias(tmp).data.cpu().numpy()]


class IPDS(Embedding):
    """ Inner Product Difference Similarity (IPDS)"""

    def __init__(self, opt):
        super(IPDS, self).__init__(opt)
        if opt["data_vectors"] is None:
            self.U = nn.Embedding(
                opt["train_node_num"],
                opt["parameter_num"] - opt["neg_dim"]
            )
            self.U_neg = nn.Embedding(
                opt["train_node_num"],
                opt["neg_dim"]
            )
        else:
            self.U = nn.Linear(opt["hidden_size"], opt[
                               "parameter_num"] - opt["neg_dim"])
            self.U_neg = nn.Linear(opt["hidden_size"], opt["neg_dim"])
        self.initialization()

    def distfn(self, input):
        u, u_neg, v, v_neg = input
        return -(torch.sum(u * v, dim=-1) - torch.sum(u_neg * v_neg, dim=-1))

    def forward(self, inputs):
        inputs = self.U_NN(self.data_vectors(inputs))
        u = self.U(inputs)
        u_neg = self.U_neg(inputs)
        ui = u.narrow(1, 1, u.size(1) - 1)
        uj = u.narrow(1, 0, 1).expand_as(ui)
        u_negi = u_neg.narrow(1, 1, u_neg.size(1) - 1)
        u_negj = u_neg.narrow(1, 0, 1).expand_as(u_negi)
        dists = self.distfn([ui, u_negi, uj, u_negj]).squeeze(-1)
        return -dists

    def embed(self):
        tmp = self.U_NN(self.data_vectors.state_dict()['weight'])
        return [self.U(tmp).data.cpu().numpy(), self.U_neg(tmp).data.cpu().numpy()]


class NPD(Embedding):
    """ Negative Poincaré Distance
    Based on the implementation of Poincaré Embedding : https://github.com/facebookresearch/poincare-embeddings
    """

    def __init__(self, opt):
        super(NPD, self).__init__(opt)
        self.dist = PDF
        if opt["data_vectors"] is None:
            self.U = nn.Embedding(
                opt["train_node_num"],
                opt["parameter_num"],
                max_norm=1
            )
        else:
            self.U = nn.Linear(opt["hidden_size"], opt["parameter_num"])
        self.initialization()

    def distfn(self, input):
        s, o = input
        print(self.dist)
        print(self.dist())        
        # return self.dist()(s, o)
        return self.dist(s, o)

    # @staticmethod
    def forward(self, inputs):
    # def forward(inputs):
        eps = 1e-5
        e = self.U(self.U_NN(self.data_vectors(inputs)))
        n = torch.norm(e, p=2, dim=2)
        mask = (n >= 1.0)
        f = n * mask.type(n.type())
        f[f != 0] /= (1.0 - eps)
        f[f == 0] = 1.0
        e = e.clone() / f.unsqueeze(2)
        o = e.narrow(1, 1, e.size(1) - 1)
        s = e.narrow(1, 0, 1).expand_as(o)
        dists = self.distfn([s, o]).squeeze(-1)
        return -dists

    def embed(self):
        eps = 1e-5
        e = self.U(self.U_NN(self.data_vectors.state_dict()['weight']))
        n = torch.norm(e, p=2, dim=1)
        mask = (n >= 1.0)
        f = n * mask.type(n.type())
        f[f != 0] /= (1.0 - eps)
        f[f == 0] = 1.0
        e = e.clone() / f.unsqueeze(1)
        return [e.data.cpu().numpy()]

class PDF(Function):
    """ Poincaré Distance Function
    Based on the implementation of Poincaré Embedding : https://github.com/facebookresearch/poincare-embeddings
    """

    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist):
        eps = 1e-5

        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) /
             torch.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = torch.sqrt(torch.pow(z, 2) - 1)
        z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    @staticmethod
    def forward(ctx, u, v):
        eps = 1e-5
        ctx.save_for_backward(u, v)
        squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        z = torch.sqrt(torch.pow(x, 2) - 1)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = PDF.grad(u, v, ctx.squnorm, ctx.sqvnorm, ctx.sqdist)
        gv = PDF.grad(v, u, ctx.sqvnorm, ctx.squnorm, ctx.sqdist)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv

# class PDF(Function):
#     """ Poincaré Distance Function
#     Based on the implementation of Poincaré Embedding : https://github.com/facebookresearch/poincare-embeddings
#     """

#     # @staticmethod
#     def grad(self, x, v, sqnormx, sqnormv, sqdist):
#         eps = 1e-5

#         alpha = (1 - sqnormx)
#         beta = (1 - sqnormv)
#         z = 1 + 2 * sqdist / (alpha * beta)
#         a = ((sqnormv - 2 * torch.sum(x * v, dim=-1) + 1) /
#              torch.pow(alpha, 2)).unsqueeze(-1).expand_as(x)
#         a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
#         z = torch.sqrt(torch.pow(z, 2) - 1)
#         z = torch.clamp(z * beta, min=eps).unsqueeze(-1)
#         return 4 * a / z.expand_as(x)

#     # @staticmethod
#     def forward(self, u, v):
#         eps = 1e-5
#         self.save_for_backward(u, v)
#         self.squnorm = torch.clamp(torch.sum(u * u, dim=-1), 0, 1 - eps)
#         self.sqvnorm = torch.clamp(torch.sum(v * v, dim=-1), 0, 1 - eps)
#         self.sqdist = torch.sum(torch.pow(u - v, 2), dim=-1)
#         x = self.sqdist / ((1 - self.squnorm) * (1 - self.sqvnorm)) * 2 + 1
#         z = torch.sqrt(torch.pow(x, 2) - 1)
#         return torch.log(x + z)

#     # @staticmethod
#     def backward(self, g):
#         u, v = self.saved_tensors
#         g = g.unsqueeze(-1)
#         gu = self.grad(u, v, self.squnorm, self.sqvnorm, self.sqdist)
#         gv = self.grad(v, u, self.sqvnorm, self.squnorm, self.sqdist)
#         return g.expand_as(gu) * gu, g.expand_as(gv) * gv
