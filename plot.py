import matplotlib.pyplot as plt
#matplotlib inline
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.linalg import expm

# Patch to 3d axis to remove margins around x, y and z limits.
# Taken from here: https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
###patch start###
from mpl_toolkits.mplot3d.axis3d import Axis
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new
###patch end###


# Rotation using euler angles from here:
# https://gist.github.com/machinaut/29d0e21b544b4a36082c761c439144d6
def rotateByEuler(points, xyz):
    ''' Rotate vector v (or array of vectors) by the euler angles xyz '''
    # https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    for theta, axis in zip(xyz, np.eye(3)):
        points = np.dot(np.array(points), expm(np.cross(np.eye(3), axis*-theta)).T)
    return points



# Fancy: plot a nice copter instead of just the flight path.
# PlotDims: dimensions of the 3d flight path plot. [0]: +- XY, [1]: + Z
def plot(results, fancy=True, plotDims=[14,30], runtime=10, framesMax=None, actionSpace=[390,440], filepath=None):

    # Set up axes grid. ###############################################################
    fig = plt.figure(figsize=(20,15))
    ax1 = plt.subplot2grid((20, 40), (0, 0), colspan=24, rowspan=20, projection='3d')
    ax2 = plt.subplot2grid((20, 40), (1, 28), colspan=12, rowspan=3)
    ax3 = plt.subplot2grid((20, 40), (5, 28), colspan=12, rowspan=3)
    ax4 = plt.subplot2grid((20, 40), (9, 28), colspan=12, rowspan=3)
    ax5 = plt.subplot2grid((20, 40), (13, 28), colspan=12, rowspan=3)
    ax6 = plt.subplot2grid((20, 40), (17, 28), colspan=12, rowspan=3)

    if framesMax == None:
        nTimesteps = len(results['x'])
    else:
        nTimesteps = framesMax


    # Plot 3d trajectory and copter. ##################################################
    c = 0.0
    plt.rcParams['grid.color'] = [c, c, c, 0.075]
    mpl.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.xmargin'] = 0

    plotLimitXY = plotDims[0]
    plotLimitZ = plotDims[1]

    quadSize = 0.5
    nPointsRotor = 15
    pointsQuadInitial = [[-quadSize, -quadSize, 0], [-quadSize, quadSize, 0], [quadSize, quadSize, 0], [quadSize, -quadSize, 0]]
    pointsRotorInitial = np.vstack(( np.sin(np.linspace(0., 2.*np.pi, nPointsRotor)),
                                     np.cos(np.linspace(0., 2.*np.pi, nPointsRotor)),
                                     np.repeat(0.0, nPointsRotor))).T * quadSize * 0.8

    # Create 3d plot.
    ax1.view_init(12, -55)
    ax1.dist = 7.6

    # Plot trajectories projected.
    xLimited = [x for x in results['x'][:nTimesteps] if np.abs(x) <= plotLimitXY]
    yLimited = [y for y in results['y'][:nTimesteps] if np.abs(y) <= plotLimitXY]
    zLimited = [z for z in results['z'][:nTimesteps] if z <= plotLimitZ]
    l = min(len(xLimited), len(yLimited))
    ax1.plot(xLimited[0:l], yLimited[0:l], np.repeat(0.0, l), c='darkgray', linewidth=0.9)
    l = min(len(xLimited), len(zLimited))
    ax1.plot(xLimited[0:l], np.repeat(plotLimitXY, l), zLimited[0:l], c='darkgray', linewidth=0.9)
    l = min(len(yLimited), len(zLimited))
    ax1.plot(np.repeat(-plotLimitXY, l), yLimited[0:l], zLimited[0:l], c='darkgray', linewidth=0.9)

    # Plot trajectory 3d.
    ax1.plot(results['x'][:nTimesteps], results['y'][:nTimesteps], results['z'][:nTimesteps], c='gray', linewidth=0.5)

    # Plot copter.
    # Colors from here: https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
    colors = np.array([ [230, 25, 75, 255],
                        [60, 180, 75, 255],
                        [255, 225, 25, 255],
                        [0, 130, 200, 255]]) / 255.

    # Plot copters and 1 second beads.
    for t in range(nTimesteps):
        # Plot copter position as dot on trajectory for each full second. ******
        if results['time'][t]%1.0 <= 0.01 or results['time'][t]%1.0 >= 0.99:
            ax1.scatter([results['x'][t]], [results['y'][t]], [results['z'][t]], s=5, c=[0., 0., 0., 0.3])
        if fancy:
            alpha1 = 0.96*np.power(t/nTimesteps, 20)+0.04
            alpha2 = 0.5 * alpha1
        else:
            alpha1 = 1.0
            alpha2 = 0.5
        # Plot frame. **********************************************************
        # Move frame.
        if fancy or t == nTimesteps -1:
            # Rotate frame.
            pointsQuad = rotateByEuler(pointsQuadInitial, np.array([results['phi'][t], results['theta'][t], results['psi'][t]]))
            # Move it.
            pointsQuad += np.array([results['x'][t], results['y'][t], results['z'][t]])
        # Plot frame projections for last time step.
        if t == nTimesteps -1:
            # Z plane.
            if np.abs(results['x'][t]) <= plotLimitXY and np.abs(results['y'][t]) <= plotLimitXY:
                ax1.plot(pointsQuad[[0,2], 0], pointsQuad[[0,2], 1], [0., 0.], c=[0., 0., 0., 0.1])
                ax1.plot(pointsQuad[[1,3], 0], pointsQuad[[1,3], 1], [0., 0.], c=[0., 0., 0., 0.1])
            # Y plane.
            if np.abs(results['x'][t]) <= plotLimitXY and np.abs(results['z'][t]) <= plotLimitZ:
                ax1.plot(pointsQuad[[0,2], 0], [plotLimitXY, plotLimitXY], pointsQuad[[0,2], 2], c=[0., 0., 0., 0.1])
                ax1.plot(pointsQuad[[1,3], 0], [plotLimitXY, plotLimitXY], pointsQuad[[1,3], 2], c=[0., 0., 0., 0.1])
            # X plane.
            if np.abs(results['y'][t]) <= plotLimitXY and np.abs(results['z'][t]) <= plotLimitZ:
                ax1.plot([-plotLimitXY, -plotLimitXY], pointsQuad[[0,2], 1], pointsQuad[[0,2], 2], c=[0., 0., 0., 0.1])
                ax1.plot([-plotLimitXY, -plotLimitXY], pointsQuad[[1,3], 1], pointsQuad[[1,3], 2], c=[0., 0., 0., 0.1])
        # Plot 3d frame for all timesteps if fancy, for last one if not.
        if fancy or t == nTimesteps -1:
            ax1.plot(pointsQuad[[0,2], 0], pointsQuad[[0,2], 1], pointsQuad[[0,2], 2], c=[0., 0., 0., alpha2])
            ax1.plot(pointsQuad[[1,3], 0], pointsQuad[[1,3], 1], pointsQuad[[1,3], 2], c=[0., 0., 0., alpha2])

        # Plot rotors. *********************************************************
        # Rotate rotor.
        if fancy or t == nTimesteps -1:
            pointsRotor = rotateByEuler(pointsRotorInitial, np.array([results['phi'][t], results['theta'][t], results['psi'][t]]))
        # Move rotor for each frame point.
        for i, color in zip(range(4), colors):
            if fancy or t == nTimesteps -1:
                pointsRotorMoved = pointsRotor + pointsQuad[i]
            # Plot rotor projections.
            if t == nTimesteps -1:
                # Z plane.
                if np.abs(results['x'][t]) <= plotLimitXY and np.abs(results['y'][t]) <= plotLimitXY:
                    ax1.add_collection3d(Poly3DCollection([list(zip(pointsRotorMoved[:,0], pointsRotorMoved[:,1], np.repeat(0, nPointsRotor)))], facecolor=[0.0, 0.0, 0.0, 0.1]))
                # Y plane.
                if np.abs(results['x'][t]) <= plotLimitXY and np.abs(results['z'][t]) <= plotLimitZ:
                    ax1.add_collection3d(Poly3DCollection([list(zip(pointsRotorMoved[:,0], np.repeat(plotLimitXY, nPointsRotor), pointsRotorMoved[:,2]))], facecolor=[0.0, 0.0, 0.0, 0.1]))
                # X plane.
                if np.abs(results['y'][t]) <= plotLimitXY and np.abs(results['z'][t]) <= plotLimitZ:
                    ax1.add_collection3d(Poly3DCollection([list(zip(np.repeat(-plotLimitXY, nPointsRotor), pointsRotorMoved[:,1], pointsRotorMoved[:,2]))], facecolor=[0.0, 0.0, 0.0, 0.1]))
            # Outline.
            if t == nTimesteps-1:
                ax1.plot(pointsRotorMoved[:,0], pointsRotorMoved[:,1], pointsRotorMoved[:,2], c=color[0:3].tolist()+[alpha1], label='Rotor {:g}'.format(i+1))
            elif fancy:
                ax1.plot(pointsRotorMoved[:,0], pointsRotorMoved[:,1], pointsRotorMoved[:,2], c=color[0:3].tolist()+[alpha1])
            # Fill.
            if fancy or t == nTimesteps -1:
                ax1.add_collection3d(Poly3DCollection([list(zip(pointsRotorMoved[:,0], pointsRotorMoved[:,1], pointsRotorMoved[:,2]))], facecolor=color[0:3].tolist()+[alpha2]))

    ax1.legend(bbox_to_anchor=(0.0 ,0.0 , 0.95, 0.85), loc='upper right')
    c = 'r'
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_xlim(-plotLimitXY, plotLimitXY)
    ax1.set_ylim(-plotLimitXY, plotLimitXY)
    ax1.set_zlim(0, plotLimitZ)
    ax1.set_xticks(np.arange(-plotLimitXY, plotLimitXY+2, 4))
    ax1.set_yticks(np.arange(-plotLimitXY, plotLimitXY+2, 4))
    ax1.set_zticks(np.arange(0, plotLimitZ+2, 4))



    # Plot velocities 2d. ###########################################################
    ax2.plot([0, runtime], [0,0], c='k', linewidth=0.5)
    ax2.plot(results['time'][:nTimesteps], results['x_velocity'][:nTimesteps], label='velocity x')
    ax2.plot(results['time'][:nTimesteps], results['y_velocity'][:nTimesteps], label='velocity y')
    ax2.plot(results['time'][:nTimesteps], results['z_velocity'][:nTimesteps], label='velocity z')
    ax2.set_ylim(-30, 30)
    ax2.set_title('Velocities')
    ax2.legend(loc=4)
    ax2.set_xticks([])
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('$[m s^{-1}]$')



    # Plot angles. ##################################################################
    ax3.plot([0, runtime], [0,0], c='k', linewidth=0.5)
    ax3.plot(results['time'][:nTimesteps], [a*180/np.pi if a<= np.pi else (a-2*np.pi)*180/np.pi for a in results['phi'][:nTimesteps]], label='phi')
    ax3.plot(results['time'][:nTimesteps], [a*180/np.pi if a<= np.pi else (a-2*np.pi)*180/np.pi for a in results['theta'][:nTimesteps]], label='theta')
    ax3.plot(results['time'][:nTimesteps], [a*180/np.pi if a<= np.pi else (a-2*np.pi)*180/np.pi for a in results['psi'][:nTimesteps]], label='psi')
    ax3.legend(loc=4)
    ax3.set_ylim(-180, 180)
    ax3.set_title('Angles')
    ax3.set_xticks([])
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('$[^{\circ}]$')



    # Plot angular velocities. ######################################################
    ax4.plot([0, runtime], [0,0], c='k', linewidth=0.5)
    ax4.plot(results['time'][:nTimesteps], [a*180/np.pi for a in results['phi_velocity'][:nTimesteps]], label='v phi')
    ax4.plot(results['time'][:nTimesteps], [a*180/np.pi for a in results['theta_velocity'][:nTimesteps]], label='v theta')
    ax4.plot(results['time'][:nTimesteps], [a*180/np.pi for a in results['psi_velocity'][:nTimesteps]], label='v psi')
    ax4.legend(loc=4)
    ax4.set_ylim(-250, 250)
    ax4.set_title('Angular velocities')
    ax4.set_xticks([])
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('$[^{\circ} s^{-1}]$')



    # Plot actions. #################################################################
    colors = np.array([ [230, 25, 75, 255],
                [60, 180, 75, 255],
                [255, 225, 25, 255],
                [0, 130, 200, 255]]) / 255.
    ax5.plot([0, runtime], [405,405], c='k', linewidth=0.5)

#    if resultsRewardMax != None:
#        ax5.plot(resultsRewardMax['time'], [a - n for a, n in zip(resultsRewardMax['rotor_speed1'], resultsRewardMax['noise_1'])], c=list(colors[0])[:3]+[0.2], label='speed 1 best')
#        ax5.plot(resultsRewardMax['time'], [a - n for a, n in zip(resultsRewardMax['rotor_speed2'], resultsRewardMax['noise_2'])], c=list(colors[1])[:3]+[0.2], label='speed 2 best')
#        ax5.plot(resultsRewardMax['time'], [a - n for a, n in zip(resultsRewardMax['rotor_speed3'], resultsRewardMax['noise_3'])], c=list(colors[2])[:3]+[0.2], label='speed 3 best')
#        ax5.plot(resultsRewardMax['time'], [a - n for a, n in zip(resultsRewardMax['rotor_speed4'], resultsRewardMax['noise_4'])], c=list(colors[3])[:3]+[0.2], label='speed 4 best')

    ax5.plot(results['time'][:nTimesteps], [a - n for a, n in zip(results['rotor_speed1'][:nTimesteps], results['noise_1'][:nTimesteps])], c=colors[0], label='speed 1')
    ax5.plot(results['time'][:nTimesteps], [a - n for a, n in zip(results['rotor_speed2'][:nTimesteps], results['noise_2'][:nTimesteps])], c=colors[1], label='speed 2')
    ax5.plot(results['time'][:nTimesteps], [a - n for a, n in zip(results['rotor_speed3'][:nTimesteps], results['noise_3'][:nTimesteps])], c=colors[2], label='speed 3')
    ax5.plot(results['time'][:nTimesteps], [a - n for a, n in zip(results['rotor_speed4'][:nTimesteps], results['noise_4'][:nTimesteps])], c=colors[3], label='speed 4')

    ax5.plot(results['time'][:nTimesteps], results['rotor_speed1'][:nTimesteps], c=colors[0], linewidth=0.5)
    ax5.plot(results['time'][:nTimesteps], results['rotor_speed2'][:nTimesteps], c=colors[1], linewidth=0.5)
    ax5.plot(results['time'][:nTimesteps], results['rotor_speed3'][:nTimesteps], c=colors[2], linewidth=0.5)
    ax5.plot(results['time'][:nTimesteps], results['rotor_speed4'][:nTimesteps], c=colors[3], linewidth=0.5)
    ax5.legend(loc=4)
    ax5.set_ylim(actionSpace[0]-3, actionSpace[1]+3)
    ax5.set_title('Rotor speeds')
    ax5.set_xticks([])
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('$[m^{-1}]$')



    # Plot episode rewards. ####################################################
    ax6.plot([0, runtime], [0,0], c='k', linewidth=0.5)
    ax6.plot(results['time'][:nTimesteps], results['reward'][:nTimesteps])
    ax6.set_ylim(-1, 1)
    ax6.set_title('Reward')
    ax6.set_xlabel('Time [s]')




    '''
    # Plot rotor speeds.
    ax2.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second', c=colors[0])
    ax2.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second', c=colors[1])
    ax2.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second', c=colors[2])
    ax2.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second', c=colors[3])
    ax2.set_ylim(0, 1000)
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('f [Hz]')
    '''
    '''
    # Plot copter angles.
    ax2.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    ax2.plot(results['time'], [a if a<= np.pi else a-2*np.pi for a in results['phi']], label='$\\alpha_x$')
    ax2.plot(results['time'], [a if a<= np.pi else a-2*np.pi for a in results['theta']], label='$\\alpha_y$')
    ax2.plot(results['time'], [a if a<= np.pi else a-2*np.pi for a in results['psi']], label='$\\alpha_z$')
    ax2.set_ylim(-np.pi, np.pi)
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('$\\alpha$ [rad]')
    ax2.legend()

    # Plot copter velocities.
    ax3.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    ax3.plot(results['time'], results['x_velocity'], label='$V_x$')
    ax3.plot(results['time'], results['y_velocity'], label='$V_y$')
    ax3.plot(results['time'], results['z_velocity'], label='$V_z$')
    ax3.set_ylim(-20, 20)
    ax3.set_xlabel('t [s]')
    ax3.set_ylabel('V [$m\,s^{1}$]')
    ax3.legend()


    # Plot copter turn rates.
    ax4.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    ax4.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    ax4.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    ax4.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    ax4.set_ylim(-3, 3)
    ax4.set_xlabel('t [s]')
    ax4.set_ylabel('$\omega$ [$rad\,s^{1}$]')
    ax4.legend()

    # Plot reward.
    ax5.plot([0,results['time'][-1]], [0,0], c=[0,0,0,0.7], linewidth=0.5)
    ax5.plot(results['time'], results['reward'], label='Reward')
    #ax5.set_ylim(-10, 10)
    ax5.set_xlabel('t [s]')
    ax5.set_ylabel('Reward')
    #ax5.legend()
    '''

    # Done :)
    plt.tight_layout()
    if filepath == None:
        plt.show()
    else:
        plt.savefig(filepath, dpi=51.2)
    plt.close(fig)
