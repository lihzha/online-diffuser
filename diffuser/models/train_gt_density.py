import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# obs = np.load("/home/lihan/diffuser-maze2d/buffer_debugging_9999.npy")[:,:,2:4].reshape((-1,2))
def train():
    gs = 10
    gs_x = 8 * gs
    gs_y = 11 * gs
    device='cuda:0'
    # grid=np.zeros((gs_x,gs_y))
    # for state in obs[...,:]:
    #     if state[0]==0.0 and state[1]==0.0:
    #         continue
    #     grid[int(np.floor(state[0]*gs_x/8)),int(np.floor(state[1]*gs_y/11))] += 1
    # np.savetxt('grid.txt',grid)
    # plt.pcolormesh(grid)
    # plt.savefig('grid.jpg')
    grid = np.loadtxt('/home/lihan/grid.txt')
    normed_grid = grid/grid.max()
    # plt.pcolormesh(normed_grid)
    # plt.savefig('normed_grid.jpg')

    class gt_density_model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2,64),
                nn.Sigmoid(),
                nn.Linear(64,128),
                nn.Sigmoid(),
                nn.Linear(128,1),
                nn.Sigmoid()
            )
        
        def forward(self,x):
            prob = self.net(x)
            return prob
        
    gt = gt_density_model().to(device)
    gt.load_state_dict(torch.load('/home/lihan/diffuser-maze2d/useful/gt_density_model/final_gt.pth'))

    def data_generator(batch_size, normed_grid):

        data_x = np.random.uniform(0.0,8.0,size=batch_size).reshape((-1,1))
        data_y = np.random.uniform(0.0,11.0,size=batch_size).reshape((-1,1))
        data = np.concatenate((data_x,data_y),axis=1)

        label = np.empty((batch_size,1))
        for i in range(batch_size):
            label[i] = normed_grid[int(np.floor(data[i][0]*gs)),int(np.floor(data[i][1]*gs))]

        return data, label

    iterations = int(10e5)
    epochs = 20
    batch_size = 1024
    opt = torch.optim.Adam(gt.parameters(),lr=1e-3)
    loss_fn = nn.MSELoss()

    def visualization(it):
        data = np.empty((gs_x,gs_y))
        for i in range(gs_x):
            for j in range(gs_y):
                inp = torch.tensor((i/gs,j/gs),dtype=torch.float,device=device)
                prob = gt(inp)
                data[i,j] = prob.detach().cpu().numpy()
        plt.pcolormesh(data)
        plt.savefig('final_result_{}.jpg'.format(it))

    for i in range(iterations):
        samples, label = data_generator(batch_size, normed_grid)
        samples = torch.tensor(samples, dtype=torch.float, device=device)
        label = torch.tensor(label, dtype=torch.float, device=device)
        for _ in range(epochs):
            pred = gt(samples)
            opt.zero_grad()
            loss = loss_fn(pred, label)
            loss.backward()
            opt.step()
        if i % 5000 == 0:
            torch.save(gt.state_dict(), 'model_weights_{}.pth'.format(i))
            visualization(i)
            print(i)

def test():
    device="cuda:0"
    gs = 100
    gs_x = 8 * gs
    gs_y = 11 * gs
    class gt_density_model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2,64),
                nn.Sigmoid(),
                nn.Linear(64,128),
                nn.Sigmoid(),
                nn.Linear(128,1),
                nn.Sigmoid()
            )
        
        def forward(self,x):
            prob = self.net(x)
            return prob
        
    gt = gt_density_model().to(device)
    gt.load_state_dict(torch.load('/home/lihan/diffuser-maze2d/useful/gt_density_model/final_gt.pth'))


    def visualization():
        data = np.empty((gs_x,gs_y))
        for i in range(gs_x):
            for j in range(gs_y):
                inp = torch.tensor((i/gs,j/gs),dtype=torch.float,device=device)
                prob = gt(inp)
                data[i,j] = prob.detach().cpu().numpy()

        plt.pcolormesh(data)
        plt.savefig('final_result.jpg')
        np.savetxt('final_gt.txt',data)
    visualization()

# train()
test()