import time
import torch
from lib import svp, svp_newton
from torchvision.io import read_image
import imageio
import numpy as np

# iterate n

torch.random.manual_seed(123)
device = torch.device("cuda:0")

im_frame = read_image('example/MITlogo.png')[:3, :]/255.
C = im_frame.shape[0]
M = im_frame.shape[1]
N = im_frame.shape[2]
gt = im_frame # 1*46*84

k = 5
p = 0.6
indices = torch.randperm(M * N)[: int(p * M * N)]  
mask = torch.zeros(M * N, dtype=torch.int32)
mask[indices] = 1
mask = mask.reshape([M, N])
observed_matrix = mask[None] * gt

imageio.imwrite(f"plot_data/svp-newton-mit-observed-{int(p * 100)}.png", (observed_matrix.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8))

observed_matrix = observed_matrix.to(device)
mask = mask.to(device)
step = 0.1
tol = 1e-3

# run svp
start = time.time()
status = {}
X_recon = svp(observed_matrix, mask, step, k, 10000, tol, status)
error = (gt - X_recon.cpu()).abs().max() / gt.abs().max()
cost = time.time() - start

print(status["iter"], cost, "s", error.item())
imageio.imwrite(f"plot_data/svp-mit-{int(p * 100)}.png", (X_recon.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8))



# run svp-Newton
start = time.time()
status = {}
X_recon = svp_newton(observed_matrix, mask, step, k, 1000, tol, status)
error = (gt - X_recon.cpu()).abs().max() / gt.abs().max()
cost = time.time() - start
print(status["iter"], cost, "s", error.item())


imageio.imwrite(f"plot_data/svp-newton-mit-{int(p * 100)}.png", (X_recon.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8))