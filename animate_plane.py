from matplotlib import pyplot as plt
from matplotlib import animation, rc, patches
import numpy as np

# Set jshtml default mode for notebook use
rc('animation', html='jshtml')

def animate(env, ctrl_pts, bc_headings, v, dt, x_ref, history, airplane=False):

    fig, ax = environment.plot_environment(env, figsize=(16, 10))
    xs = x_ref[0, :]
    ys = x_ref[1, :]
    x0s = [history[i]['x0'][0] for i in history.keys()]
    y0s = [history[i]['x0'][1] for i in history.keys()]

    # plot reference trajectory
    ax.plot(xs, ys, '-o', alpha=0.8, markersize=3, color='blue', label='reference trajectory')
    # optimized trajectory
    opt_line, = ax.plot([], [], '-o', lw=3, color='lightseagreen', label='N-step optimal traj')
    nom_line, = ax.plot([], [], color='red', label='nominal trajectory')
    act_line, = ax.plot([], [], '-o', lw=3, markersize=6, color='blueviolet', label='actual trajectory')

    if airplane:
        # Define the airplane polygon (a simple representation)
        airplane_shape = np.array([
            [0, 0],  # Tail
            [1, 0],  # Tail to Body
            [1, 0.5],  # Body to Wing
            [2, 0],  # Wing Tip
            [1, -0.5],  # Wing to Body
            [1, 0],  # Body to Nose
            [3, 0],  # Nose
            [1, 0],  # Nose to Body
            [1, 0.5],  # Body to Opposite Wing
            [2, 0],  # Wing Tip
            [1, -0.5],  # Wing to Body
            [1, 0],  # Body to Tail
        ])
        airplane_patch = patches.Polygon(airplane_shape, closed=True, fill=None, edgecolor='k')

    def init():
        opt_line.set_data([], [])
        act_line.set_data([], [])
        nom_line.set_data([], [])
        if airplane:
            ax.add_patch(airplane_patch)
        return opt_line,

    def step(i):
        x = history[i]['x_opt'][0]
        y = history[i]['x_opt'][1]
        xbar = history[i]['x_bar'][0]
        ybar = history[i]['x_bar'][1]
        opt_line.set_data(x, y)
        act_line.set_data(x0s[:i], y0s[:i])
        nom_line.set_data(xbar, ybar)
        if airplane:
            if len(x) > 1:
                heading = np.arctan2((y[1] - y[0]), (x[1] - x[0]))
            else:
                heading = bc_headings[1]
            transform = patches.Affine2D().rotate_around(airplane_shape[0][0], airplane_shape[0][1], heading)
            airplane_patch.set_transform(transform + ax.transData)
            airplane_patch.set_xy((x[0], y[0]))

        return opt_line,

    anim = animation.FuncAnimation(fig, step, init_func=init, frames=len(history.keys()), interval=1000*dt*2, blit=True)

    ax.axis('equal')
    ax.legend()
    plt.close()
    return anim

if __name__ == "__main__":

    # TEST ANIMATION FUNCTION

    scene = scenarios.five_obstacle
    ctrl_pts = scene['control_pts']
    bch = scene['bc_headings']
    env = environment.Environment(scene['obs_list'], scene['start'], scene['goal'])
    env.add_control_points(ctrl_pts)
    v = 4
    dt = 0.1
    xs, ys, psi = trajectory_gen.sample_trajectory(ctrl_pts, bch, v, dt)
    nf = len(xs)
    x_ref = np.vstack((xs.reshape((1, nf)),
                       ys.reshape((1, nf)),
                       psi.reshape((1, nf)),
                       v * np.ones((1, nf)),
                       np.zeros((1, nf))))

    # Call the animate function with the airplane shape
    anim = animate(env, ctrl_pts, bch, v, dt, x_ref, None, airplane=True)
    plt.show()
