import torch
from lib import svp, svp_newton
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

torch.random.manual_seed(123)

im_frame = Image.open('example/MITlogo.png')
np_frame = np.array(im_frame.getdata())
M = 46
N = 81
np_frame = np_frame.reshape((M,N,4))
np_frame = np_frame[:,:,0]
gt = torch.from_numpy(np_frame[None]).double() # 1*46*84

C = 1
rank = 5

indices = torch.randperm(M * N)[: int(0.6 * M * N)]  # 20% elements are observed
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