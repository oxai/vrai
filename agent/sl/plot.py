import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib as matlib
%matplotlib

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

f = open("generated.txt","r")
coords = np.array([[0,0,0]])
coords.shape
num_repeats = 1
n=10
for i,line in enumerate(f.readlines()):
    if line[0] != "[":
        num_repeats = int(line[:-1])
    else:
        line = line[1:-2]
        linearr = line.split(";")
        arr = np.array([[float(x) for x in linearr]])
        if i == 1:
            coords = arr
        else:
            coords = np.concatenate([coords,matlib.repmat(arr,num_repeats,1)],axis=0)

coords.shape

# ax.plot(coords[:,0],coords[:,1],coords[:,2], label='parametric curve')

line, = ax.plot(coords[:,0],coords[:,1],coords[:,2], label='parametric curve')
#%%
# def update(t, line):
#     print(t)
#     line.set_data(coords[:t,0],coords[:t,1],coords[:t,2])
#     return line,

# import matplotlib.animation as animation
# ani = animation.FuncAnimation(fig, update, len(coords), fargs=[line],
#                               interval=25, blit=True)
# ax.plot([:,0],coords[:,1],coords[:,2], label='parametric curve')
# ax.legend()
ax.cla()
import time
t=1
for t in range(len(ycoord)):
    # ax.plot(obs[:,0],obs[:,1],obs[:,2], label='parametric curve')
    ax.plot(coords[t:t+2,0],coords[t:t+2,1],coords[t:t+2,2], label='parametric curve',c=(0,0,0,0.2))
    # ax.plot(xcoord[t:t+2],ycoord[t:t+2],zcoord[t:t+2], label='parametric curve',c=(0,0,1,0.2))
    # t+=1
    # ax.legend()

    # plt.show()
    plt.savefig("figs/{0:03d}.png".format(t))
    # time.sleep(3)
    # ax.cla()

# plt.show()


#%%
obs = np.load("circling_box_obs.npy")
n=10
discretized_obs = np.sum(np.floor(((obs+1)/2)*n)*np.array([1,n,n**2]),axis=1)
change_times = (np.where(np.diff(discretized_obs)>0)[0]) #max change time is 2000
states = discretized_obs[change_times]
change_times = n**3+change_times//100
# len(states)
# combined = np.stack([change_times,states],axis=1)
# change_times[:10]
# states[:10]
# combined = combined.flatten()
combined = states
np.max(combined)

#%%
# n**3
fig = plt.figure()
ax = fig.gca(projection='3d')
# n=20
# discretized_obs = np.sum(np.floor(((obs+1)/2)*n)*np.array([1,n,n**2]),axis=1)
# discretized_obs += 1000
# discretized_obs-=1000
# zcoord[200]
xcoord = states//n**2
ycoord = (states%n**2)//n
zcoord = states%n
#
# import time
# for t in range(len(xcoord)):
#     ax.plot(2*xcoord[:t]/n-1,2*ycoord[:t]/n-1,2*zcoord[:t]/n-1, label='parametric curve')
#     # ax.plot(obs[:,0],obs[:,1],obs[:,2], label='parametric curve')
#     # ax.legend()
#
#     plt.show()
#     time.sleep(0.5)

# obs
