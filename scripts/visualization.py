import matplotlib.pyplot as plt
import numpy as np
gs=10
gs_x=8*gs
gs_y=11*gs
obs = np.load("/home/lihan/diffuser-maze2d/logs/maze2d-large-v1/diffusion/4_13_ebm_scratch_pexplore0.5/buffer_vis.npy")[:,:,:2].reshape((-1,2))
grid=np.zeros((gs_x,gs_y))
for state in obs[...,:]:
    if state[0]==0.0 and state[1]==0.0:
        continue
    grid[int(np.floor(state[0]*gs_x/8)),int(np.floor(state[1]*gs_y/11))] += 1
# np.savetxt('grid.txt',grid)
plt.pcolormesh(grid)
plt.savefig('/home/lihan/diffuser-maze2d/grid1457ebm.jpg')