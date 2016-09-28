import numpy as np
from scipy import integrate

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation

from time import sleep

N_trajectories = 4

S = 10.
B = 8./3
def lorentz_deriv(X, t0, sigma=S, beta=B, rho=S * (S + B + 3) / (S - B - 1)):
    """Compute the time-derivative of a Lorentz system."""
    x, y, z = X
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


# Choose random starting points, uniformly distributed from -15 to 15
np.random.seed(1)
center = (0., 0., 25.)
distance = 20
direction = [
    [1., 0., 0.],
    [0., 1., 0.],
    [2**(-.5), 0., 2**(-.5)],
    [0., 2.**(-.5), 2.**(-5)],
]
x0 = distance * np.array(direction) + center

# Solve for the trajectories
fps = 25
video_time = 50
T = 50.
et = -5
dt = 10**et
# I = .01/dt
I = (T/dt) / (fps * video_time)
t = np.linspace(0., T, int(T/dt))
x_t = np.asarray([integrate.odeint(lorentz_deriv, x0i, t)
                  for x0i in x0])

# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.axis('off')

# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

# prepare the axes limits
ax.set_xlim((-20, 20))
ax.set_ylim((-20, 20))
ax.set_zlim((5, 45))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = int((I * i) % x_t.shape[1])

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(30, 2 * i / I)
    fig.canvas.draw()
    return lines + pts

# instantiate the animator.

# sleep(20)
# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(T/(dt*I)),
                               interval=30, blit=True)
anim.save('E%d.%ds.mp4'%(et, video_time), writer=writer)

# Save as mp4. This requires mplayer or ffmpeg to be installed
#anim.save('lorentz_attractor.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

# plt.show()
