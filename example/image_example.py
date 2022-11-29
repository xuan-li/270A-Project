import torch
from lib import svp, svp_newton
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.io import read_image

torch.random.manual_seed(123)

im_frame = read_image('example/MITlogo.png')[:3, :]/255.
C = im_frame.shape[0]
M = im_frame.shape[1]
N = im_frame.shape[2]
gt = im_frame # 1*46*84

rank = 5

indices = torch.randperm(M * N)[: int(0.6 * M * N)]  
mask = torch.zeros(M * N, dtype=torch.int32)
mask[indices] = 1
mask = mask.reshape([M, N])
observed_matrix = mask[None] * gt

problem = dict(
    channel = 1,
    size=[M, N],
    gt=gt,
    rank=rank,
    observed_matrix=observed_matrix,
    mask=mask,
    method=svp_newton,
    # method=svp,
    step=1,
    tol=1e-4,
    device="cuda:0"
)

del(torch)