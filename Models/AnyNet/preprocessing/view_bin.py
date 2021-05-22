import mayavi.mlab
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Generate Libar')
parser.add_argument('--image', type=str,
                    default='0')
args = parser.parse_args()
dirname = os.path.dirname(__file__)


pointcloud = np.fromfile(os.path.join(dirname, args.image.zfill(6)+".bin"), dtype=np.float32, count=-1).reshape(-1,4)

print(pointcloud)
print(pointcloud.shape)
x = pointcloud[:, 0]  # x position of point
y = pointcloud[:, 1]  # y position of point
z = pointcloud[:, 2]  # z position of point
r = pointcloud[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

vals='height'
if vals == "height":
    col = z
else:
    col = d
 
fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
mayavi.mlab.points3d(x, y, z,
                    col,          # Values used for Color
                    mode="point",
                    colormap='spectral', # 'bone', 'copper', 'gnuplot'
                    # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                    figure=fig,
                    )
 
x=np.linspace(5,5,50)
y=np.linspace(0,0,50)
z=np.linspace(0,5,50)
mayavi.mlab.plot3d(x,y,z)
mayavi.mlab.show()
