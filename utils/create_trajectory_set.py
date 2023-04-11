import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from random import randint


def find_representative_trajectory(pred_trajs):

    '''
    pred_trajs : num_trajs x seq_len x 2
    '''

    num_trajs, seq_len, dim = pred_trajs.shape

    distances = []
    for k in range(num_trajs):
        cur_pred_traj_repeat = np.repeat(np.expand_dims(pred_trajs[k], axis=0), num_trajs, axis=0)
        err = cur_pred_traj_repeat - pred_trajs  # num_trajs x seq_len x dim

        dist = np.sum(np.sqrt(np.sum(err ** 2, axis=2)))
        distances.append(dist)

    minidx = np.argmin(np.array(distances))

    return pred_trajs[minidx].reshape(1, seq_len, dim)

def FixedTrajectorySetGenerator(Kp, epsilon):

    '''
    Implementation of equation (1) in "CoverNet: Multimodal behavior prediction using trajectory sets, CVPR 2020"

    <Input>
    Kp : nparray of size N x seq_len x dim
    epsilon : variable (meter)

    <Output>
    K : nparray of size M x seq_len x dim, M <= N
    '''

    start_time = time.time()
    # code starts here -------

    Kp_copy = np.copy(Kp)
    K = []
    while (Kp_copy.size > 0):

        N, seq_len, dim = Kp_copy.shape

        # randomly pick a trajectory from Kp
        idx = randint(0, N-1)
        cur_traj = np.expand_dims(Kp_copy[idx], axis=0) # 1 x seq_len x dim

        # calc distance
        error = Kp_copy - np.repeat(cur_traj, N, axis=0) # N x seq_len x dim
        error = np.sqrt(np.sum(error ** 2, axis=2)) # N x seq_len
        error_max = np.amax(error, axis=1)

        # find trajectories whose error is larger than epsilon
        chk = error_max > epsilon
        Kp_comp = Kp_copy[~chk]
        Kp_copy = Kp_copy[chk]

        # find the representative traj
        if (Kp_comp.size > 0):
            cur_group = np.concatenate([cur_traj, Kp_comp], axis=0)
            repr_traj = find_representative_trajectory(cur_group)
        else:
            repr_traj = cur_traj

        # insert current trajectory into K
        K.append(repr_traj)

    K = np.concatenate(K, axis=0)

    # code ends here -------
    end_time = time.time()

    return K, (end_time - start_time)



file_path = './trajectories.pkl'


sample_period = 2 # Hz
target_pred_horizon = 4 # second
pred_len = sample_period * target_pred_horizon
epsilon = 2.0

with open(file_path, 'rb') as f:
    trajectories = pickle.load(f)

K, time_spent = FixedTrajectorySetGenerator(trajectories[:, :pred_len, :], epsilon)

# plot trajectories
plt.plot(-1.0*K[:, :, 1].T, K[:, :, 0].T)
plt.axis([-80, 80, -40, 120])
plt.title("epsilon: %.2f m, num trajs in K: %d, time spent: %.4f sec" % (epsilon, K.shape[0], time_spent))
plt.xlabel('y-axis')
plt.ylabel('x-axis')
plt.show()





