#%%
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def rewardHeightPunishPowerLinear(w_x, w_p, x_t, x_min, x_max, p_t, p_min, p_max):
  R_x_t = (x_t - x_min) / (x_max - x_min)
  R_p_t = (p_t - p_max) / (p_min - p_max)

  R_t = w_x * R_x_t + w_p * R_p_t
  R_min = 0
  R_max = w_x + w_p

  R_t_norm = (R_t - R_min) / (R_max - R_min)

  return R_t_norm

def rewardHeightPunishPowerNonlinear(w_x, w_p, x_t, x_min, x_max, p_t, p_min, p_max):
  index_1 = 0
  for pos1 in x_t:
    index_2 = 0
    for pos2 in pos1:
      if pos2 > x_max:
        x_t[index_1, index_2] = x_max
      index_2 = index_2 + 1
    index_1 = index_1 + 1

  R_x_t = (x_t - x_min) / (x_max - x_min)
  R_p_t = (p_t - p_max) / (p_min - p_max)

  R_t = w_x * R_x_t**(3) + w_p * R_p_t**(1/3)
  R_min = 0 
  R_max = w_x + w_p

  R_t_norm = (R_t - R_min) / (R_max - R_min)

  return R_t_norm

#%% Linear Reward
      
w_xs = np.arange(.1, 1, .1)
columns = 3
rows = 3

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=(15,15))

index = 1
for y in range(rows):
    for x in range(columns):
        # set up the axes for the first plot
        ax = fig.add_subplot(rows, columns, index, projection='3d')

        w_x = w_xs[index - 1]
        w_p = 1 - w_x
        x_min = 0
        x_max = 0.25
        p_min = 0
        p_max = 10

        x_t = np.linspace(0, x_max, 100)
        p_t = np.linspace(1, p_max, 100)


        # fig = plt.figure()
        # ax = Axes3D(fig)
        x_t, p_t = np.meshgrid(x_t, p_t)
        r_t = rewardHeightPunishPowerLinear(w_x=w_x, w_p=w_p, x_t=x_t, x_min=x_min, x_max=x_max, p_t=p_t, p_min=p_min, p_max=p_max)
        
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        ax.plot_surface(x_t, p_t, r_t, rstride=1, cstride=1, cmap='hot')
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
            item.set_fontsize(15)

        ax.set_title(f'$\omega_x$ = {w_x}', size=20, y=1)
        index = index + 1
        
ax.set_xlabel('Jump Height')
ax.set_ylabel('Power Used')
ax.set_zlabel('Reward')

fig.tight_layout()
file_name = f'RewardHeightPunishPowerLinear.png'
plt.savefig(file_name)
plt.show()

#%% Nonlinear Reward

w_xs = np.arange(.1, 1, .1)
columns = 3
rows = 3

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=(15,15))

index = 1
for y in range(rows):
    for x in range(columns):
        # set up the axes for the first plot
        ax = fig.add_subplot(rows, columns, index, projection='3d')

        w_x = w_xs[index - 1]
        w_p = 1 - w_x
        x_min = 0
        x_max = 0.25
        p_min = 0
        p_max = 10

        x_t = np.linspace(0, x_max, 100)
        p_t = np.linspace(0, p_max, 100)


        # fig = plt.figure()
        # ax = Axes3D(fig)
        x_t, p_t = np.meshgrid(x_t, p_t)
        r_t = rewardHeightPunishPowerNonlinear(w_x=w_x, w_p=w_p, x_t=x_t, x_min=x_min, x_max=x_max, p_t=p_t, p_min=p_min, p_max=p_max)

        ax.invert_xaxis()
        ax.invert_yaxis()
        
        ax.plot_surface(x_t, p_t, r_t, rstride=1, cstride=1, cmap='hot')
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
            item.set_fontsize(15)

        ax.set_title(f'$\omega_x$ = {w_x}', size=20, y=1)
        index = index + 1
        
ax.set_xlabel('Jump Height')
ax.set_ylabel('Power Used')
ax.set_zlabel('Reward')

fig.tight_layout()
file_name = f'RewardHeightPunishPowerNonlinear.png'
plt.savefig(file_name)
plt.show()
# %%
