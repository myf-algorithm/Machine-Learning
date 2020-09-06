import torch


def PageRank(R0, d, M, iters):
    dM = d * M
    smooth = (1 - d) * torch.ones((len(M), 1)).fill_(1 / len(M))

    for _ in range(iters):
        R0 = torch.matmul(dM, R0) + smooth
    return R0


M = torch.tensor([[0, 1 / 2, 0, 0],
                  [1 / 3, 0, 0, 1 / 2],
                  [1 / 3, 0, 1, 1 / 2],
                  [1 / 3, 1 / 2, 0, 0],
                  ])

R0 = torch.ones((4, 1)).fill_(1 / 4)
print(PageRank(R0, 0.8, M, 100000))
