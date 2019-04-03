from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

n_angles=36
n_radii=8
radii=np.linspace(0.125,1,n_radii)
#在指定的间隔内返回均匀间隔的数字。
#numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None),endpoint=True表示包括终止点的数。
angles=np.linspace(0,2*np.pi,n_angles,endpoint=False)
angles=np.repeat(angles[:,np.newaxis],n_radii,axis=1)
#将上一个angle的数组生成一个多维的数组，之后生成重复的n_radii个，沿着x轴复制。
x=np.append(0,(radii*np.cos(angles)).flatten())#降到一维
y=np.append(0,(radii*np.sin(angles)).flatten())
z=np.sin(-x*y)
fig=plt.figure()
ax=fig.gca(projection='3d')
ax.plot_trisurf(x,y,z,cmap=cm.jet,linewidth=0.9)#生成曲面图形
plt.show()