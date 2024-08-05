import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation, rc
from matplotlib.patches import Rectangle
import environment
import scenarios
import trajectory_gen

# Set jshtml default mode for notebook use
rc('animation', html='jshtml')

def plot_one_step(env, x_ref, x_bar, x_opt, x_next=None, nominals=None):
    ''' Plots a single step of trajectory optimization in environment '''
    fig, ax = environment.plot_environment(env, figsize=(16, 10))
    ax.plot(x_ref[0, :], x_ref[1, :], '-o', alpha=0.8,
            color='blue', markersize=3, label='reference trajectory')
    if nominals is not None:
        for nom in nominals:
            ax.plot(nom[0, :], nom[1, :], 'r-', label='nominal trajectories')
    else:
        ax.plot(x_bar[0, :], x_bar[1, :], 'r-', label='nominal trajectory')
    ax.plot(x_opt[0, :], x_opt[1, :], '-o',
            color='lightseagreen', label='optimal trajectory')
    if x_next is not None:
        ax.plot(x_next[0], x_next[1], 'o', color='blueviolet', label='next x0')

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels, loc='best')
    return fig, ax

def plot_all_steps(env, x_ref_full, history):
    ''' Plots optimization paths in environment at each step over time
        course of the full MPC run
    '''
    fig, ax = environment.plot_environment(env, figsize=(16, 10))
    ax.plot(x_ref_full[0, :], x_ref_full[1, :],
            '-o', color='blue', label='reference trajectory', markersize=3)
    for i in range(len(history.keys())):
        xi = history[i]['x_opt']
        ax.plot(xi[0, :], xi[1, :], color='lightseagreen',
                linewidth=1, label='N-step optimal traj')
    x0x = [history[i]['x0'][0] for i in history.keys()]
    x0y = [history[i]['x0'][1] for i in history.keys()]
    ax.plot(x0x, x0y, '-o', color='blueviolet', label='actual trajectory')
    xf_bar = history[len(history.keys())-1]['x_bar']
    # ax.plot(xf_bar[0, :], xf_bar[1, :], 'r', label='xf_bar')
    ax.axis('equal')
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels, loc='best')
    return fig, ax

def animate(env, ctrl_pts, bc_headings, v, dt, x_ref, history, shape='rect'):

    fig, ax = environment.plot_environment(env, figsize=(16, 10))
    xs = x_ref[0, :]
    ys = x_ref[1, :]
    x0s = [history[i]['x0'][0] for i in history.keys()]
    y0s = [history[i]['x0'][1] for i in history.keys()]

    # plot reference trajectory
    ax.plot(xs, ys, '-o', alpha=0.8, markersize=3,
            color='blue', label='reference trajectory')
    # optimized trajectory
    opt_line, = ax.plot([], [], '-o', lw=3, color='lightseagreen',
                        label='N-step optimal traj')
    nom_line, = ax.plot([], [], color='red', label='nominal trajectory')
    act_line, = ax.plot([], [], '-o', lw=3, markersize=6,
                        color='blueviolet', label='actual trajectory')

    def draw_airplane(ax, position=(0, 0), heading=0, scale=1.0):
        # Define the dimensions of the airplane components
        fuselage_h, fuselage_w = 0.5 * scale, 0.2 * scale
        wing_h, wing_w = 0.1 * scale, 0.2 * scale
        tail_h, tail_w = 0.1 * scale, 0.2 * scale

        # Calculate the rotation matrix
        R = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])

        # Define the positions of each component relative to the fuselage center
        fuselage_pos = np.array([0, 0])
        left_wing_pos = np.array([-(fuselage_w/2+wing_w), -wing_h/2])
        right_wing_pos = np.array([fuselage_w/2, -wing_h/2])
        tail_pos = np.array([-tail_w/2, -(fuselage_h/2+tail_h/2)])

        # Rotate and translate each component
        def transform(pos):
            return np.dot(R, pos) + np.array(position)

        fuselage_pos = transform(fuselage_pos) - np.array([fuselage_w / 2, fuselage_h / 2])
        left_wing_pos = transform(left_wing_pos)
        right_wing_pos = transform(right_wing_pos)
        tail_pos = transform(tail_pos)

        # Create and add the rectangles
        fuselage = Rectangle(fuselage_pos, fuselage_w, fuselage_h, angle=np.degrees(heading), rotation_point='center', edgecolor='k', fill=None)
        left_wing = Rectangle(left_wing_pos, wing_w, wing_h, angle=np.degrees(heading), rotation_point='center', edgecolor='k', fill=None)
        right_wing = Rectangle(right_wing_pos, wing_w, wing_h, angle=np.degrees(heading), rotation_point='center', edgecolor='k', fill=None)
        tail = Rectangle(tail_pos, tail_w, tail_h, angle=np.degrees(heading), rotation_point='center', edgecolor='k', fill=None)

        # Add the components to the plot
        ax.add_patch(fuselage)
        ax.add_patch(left_wing)
        ax.add_patch(right_wing)
        ax.add_patch(tail)

    if shape == 'rect':
        ld, wd = 0.5, 0.2
        a2 = np.arctan2(wd, ld)
        diag = np.sqrt(ld**2 + wd**2)
        heading = np.rad2deg(np.arctan2((y0s[1]-y0s[0]), (x0s[1]-x0s[0])))
        vehicle = Rectangle(
            (x0s[0]-ld, y0s[0]-wd), 2*ld, 2*wd, angle=-heading, fc='none', lw=1, ec='k')
    elif shape == 'airplane':
        heading = np.arctan2((y0s[1]-y0s[0]), (x0s[1]-x0s[0]))
        draw_airplane(ax, position=(x0s[0], y0s[0]), heading=heading, scale=1.0)

    def init():
        opt_line.set_data([], [])
        act_line.set_data([], [])
        nom_line.set_data([], [])
        return opt_line,

    def step(i):
        x = history[i]['x_opt'][0]
        y = history[i]['x_opt'][1]
        xbar = history[i]['x_bar'][0]
        ybar = history[i]['x_bar'][1]
        opt_line.set_data(x, y)
        act_line.set_data(x0s[:i], y0s[:i])
        nom_line.set_data(xbar, ybar)
        if shape == 'rect':
            if (len(x) == 1):
                heading = bc_headings[1]
            else:
                heading = np.arctan2((y[1]-y[0]), (x[1]-x[0]))
            xoff = diag*np.cos(heading + a2)
            yoff = diag*np.sin(heading + a2)
            vehicle.set_xy((x[0] - xoff, y[0] - yoff))
            vehicle.angle = np.rad2deg(heading)
        elif shape == 'airplane':
            ax.patches = []  # Clear the previous patches
            heading = np.arctan2((y[1]-y[0]), (x[1]-x[0]))
            draw_airplane(ax, position=(x[0], y[0]), heading=heading, scale=1.0)

        return opt_line,

    anim = animation.FuncAnimation(fig, step, init_func=init,
                                   frames=len(history.keys()), interval=1000*dt*2, blit=True)

    ax.axis('equal')
    ax.legend()
    plt.close()
    return anim

if __name__ == "__main__":

    # TEST ANIMATION FUNCTION

    scene = scenarios.five_obstacle
    ctrl_pts = scene['control_pts']
    bch = scene['bc_headings']
    env = environment.Environment(scene['obs_list'],
                                  scene['start'], scene['goal'])
    env.add_control_points(ctrl_pts)
    v = 4
    dt = 0.1
    xs, ys, psi = trajectory_gen.sample_trajectory(ctrl_pts, bch, v, dt)
    nf = len(xs)
    x_ref = np.vstack((xs.reshape((1, nf)),
                       ys.reshape((1, nf)),
                       psi.reshape((1, nf)),
                       v*np.ones((1, nf)),
                       np.zeros((1, nf))))

    anim = animate(env, ctrl_pts, bch, v, dt, x_ref, None, shape='airplane')
    plt.show()
