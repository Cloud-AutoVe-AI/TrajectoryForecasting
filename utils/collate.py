from utils.libraries import *
from utils.functions import toNP, toTS

def seq_collate_typeA(data):

    obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, \
    R_map, R_traj, num_agents, valid_loss_flag = [], [], [], [], [], [], [], [], []

    _obs_traj, _future_traj, _obs_traj_e, _future_traj_e, _feature_topview, \
    _R_map, _R_traj, _num_agents, _valid_loss_flag = zip(*data)

    valid_index = []
    for idx, cnt in enumerate(_num_agents):
        if (cnt > 1):
            obs_traj.append(_obs_traj[idx])
            future_traj.append(_future_traj[idx])
            obs_traj_e.append(_obs_traj_e[idx])
            future_traj_e.append(_future_traj_e[idx])
            feature_topview.append(_feature_topview[idx])
            R_map.append(_R_map[idx])
            R_traj.append(_R_traj[idx])
            num_agents.append(_num_agents[idx])
            valid_loss_flag.append(_valid_loss_flag[idx])

            valid_index.append(idx)

    if (len(valid_index) == 0):
        obs_traj = _obs_traj
        future_traj = _future_traj
        obs_traj_e = _obs_traj_e
        future_traj_e = _future_traj_e
        feature_topview = _feature_topview
        R_map = _R_map
        R_traj = _R_traj
        num_agents = _num_agents
        valid_loss_flag = _valid_loss_flag
    else:
        num_repeat = len(_num_agents) - len(valid_index)
        for i in range(num_repeat):
            idx = random.randint(0, len(valid_index)-1)
            obs_traj.append(_obs_traj[valid_index[idx]])
            future_traj.append(_future_traj[valid_index[idx]])
            obs_traj_e.append(_obs_traj_e[valid_index[idx]])
            future_traj_e.append(_future_traj_e[valid_index[idx]])
            feature_topview.append(_feature_topview[valid_index[idx]])
            R_map.append(_R_map[valid_index[idx]])
            R_traj.append(_R_traj[valid_index[idx]])
            num_agents.append(_num_agents[valid_index[idx]])
            valid_loss_flag.append(_valid_loss_flag[valid_index[idx]])


    _len = [objs for objs in num_agents]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj_cat = torch.cat(obs_traj, dim=1)
    future_traj_cat = torch.cat(future_traj, dim=1)
    obs_traj_cat_e = torch.cat(obs_traj_e, dim=1)
    future_traj_cat_e = torch.cat(future_traj_e, dim=1)
    feature_topview_cat = torch.cat(feature_topview, dim=0)
    R_map_cat = torch.cat(R_map, dim=0)
    R_traj_cat = torch.cat(R_traj, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    valid_loss_flag_cat = np.concatenate(valid_loss_flag, axis=1)

    return tuple([obs_traj_cat, future_traj_cat, obs_traj_cat_e, future_traj_cat_e,
                  feature_topview_cat, R_map_cat, R_traj_cat, seq_start_end, num_agents, valid_loss_flag_cat])

def seq_collate_typeB(data):


    _future_traj_e, _feature_topview, _agent_states, _pred_len = zip(*data)
    future_traj_e, feature_topview, agent_states, num_agents, valid_index= [], [], [], [], []

    num_agents = len(_agent_states)
    for idx in range(num_agents):

        if (np.count_nonzero(np.isnan(_agent_states[idx])) > 0):
            continue

        if (_future_traj_e[idx].shape[0] != _pred_len[0]):
            continue

        future_traj_e.append(_future_traj_e[idx])
        feature_topview.append(_feature_topview[idx])
        agent_states.append(_agent_states[idx])

    num_valid_agents = len(agent_states)
    if (num_valid_agents == 0):
        return tuple([np.nan, np.nan, np.nan])


    num_repeat = num_agents - num_valid_agents
    for i in range(num_repeat):
        idx = random.randint(0, num_valid_agents-1)
        future_traj_e.append(future_traj_e[idx])
        feature_topview.append(feature_topview[idx])
        agent_states.append(agent_states[idx])

    future_traj_e_cat = torch.cat(future_traj_e, dim=1)
    feature_topview_cat = torch.cat(feature_topview, dim=0)
    agent_states_cat = torch.cat(agent_states, dim=0)

    return tuple([future_traj_e_cat, feature_topview_cat, agent_states_cat])

def seq_collate_typeCps(data):


    obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, \
    map, num_neighbors, valid_neighbor, possible_path, lane_label = zip(*data)

    _len = [objs for objs in num_neighbors]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj_ta_cat = torch.cat(obs_traj_ta, dim=1)
    future_traj_ta_cat = torch.cat(future_traj_ta, dim=1)
    obs_traj_ngh_cat = torch.cat(obs_traj_ngh, dim=1)
    future_traj_ngh_cat = torch.cat(future_traj_ngh, dim=1)

    map_cat = torch.cat(map, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    possible_path_cat = torch.cat(possible_path, dim=1)
    lane_label_cat = torch.cat(lane_label, dim=0)

    return tuple([obs_traj_ta_cat, future_traj_ta_cat, obs_traj_ngh_cat, future_traj_ngh_cat,
                  map_cat, seq_start_end, np.array(valid_neighbor), possible_path_cat, lane_label_cat])

def seq_collate_typeD(data):

    obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, \
    R_map, R_traj, num_agents = [], [], [], [], [], [], [], []

    _obs_traj, _future_traj, _obs_traj_e, _future_traj_e, _feature_topview, \
    _R_map, _R_traj, _num_agents = zip(*data)

    valid_index = []
    for idx, cnt in enumerate(_num_agents):
        if (cnt > 1):
            obs_traj.append(_obs_traj[idx])
            future_traj.append(_future_traj[idx])
            obs_traj_e.append(_obs_traj_e[idx])
            future_traj_e.append(_future_traj_e[idx])
            feature_topview.append(_feature_topview[idx])
            R_map.append(_R_map[idx])
            R_traj.append(_R_traj[idx])
            num_agents.append(_num_agents[idx])

            valid_index.append(idx)

    if (len(valid_index) == 0):
        obs_traj = _obs_traj
        future_traj = _future_traj
        obs_traj_e = _obs_traj_e
        future_traj_e = _future_traj_e
        feature_topview = _feature_topview
        R_map = _R_map
        R_traj = _R_traj
        num_agents = _num_agents
    else:
        num_repeat = len(_num_agents) - len(valid_index)
        for i in range(num_repeat):
            idx = random.randint(0, len(valid_index)-1)
            obs_traj.append(_obs_traj[valid_index[idx]])
            future_traj.append(_future_traj[valid_index[idx]])
            obs_traj_e.append(_obs_traj_e[valid_index[idx]])
            future_traj_e.append(_future_traj_e[valid_index[idx]])
            feature_topview.append(_feature_topview[valid_index[idx]])
            R_map.append(_R_map[valid_index[idx]])
            R_traj.append(_R_traj[valid_index[idx]])
            num_agents.append(_num_agents[valid_index[idx]])

    _len = [objs for objs in num_agents]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj_cat = torch.cat(obs_traj, dim=1)
    future_traj_cat = torch.cat(future_traj, dim=1)
    obs_traj_cat_e = torch.cat(obs_traj_e, dim=1)
    future_traj_cat_e = torch.cat(future_traj_e, dim=1)
    feature_topview_cat = torch.cat(feature_topview, dim=0)
    R_map_cat = torch.cat(R_map, dim=0)
    R_traj_cat = torch.cat(R_traj, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    return tuple([obs_traj_cat, future_traj_cat, obs_traj_cat_e, future_traj_cat_e,
                  feature_topview_cat, R_map_cat, R_traj_cat, seq_start_end, num_agents])

def seq_collate_typeE(data):

    obs_traj, future_traj, obs_traj_e, future_traj_e, feature_topview, \
    R_map, R_traj, num_agents = [], [], [], [], [], [], [], []

    _obs_traj, _future_traj, _obs_traj_e, _future_traj_e, _feature_topview, \
    _R_map, _R_traj, _num_agents = zip(*data)

    valid_index = []
    for idx, cnt in enumerate(_num_agents):
        if (np.count_nonzero(np.isnan(_future_traj[idx])) == 0):
            obs_traj.append(_obs_traj[idx])
            future_traj.append(_future_traj[idx])
            obs_traj_e.append(_obs_traj_e[idx])
            future_traj_e.append(_future_traj_e[idx])
            feature_topview.append(_feature_topview[idx])
            R_map.append(_R_map[idx])
            R_traj.append(_R_traj[idx])
            num_agents.append(_num_agents[idx])

            valid_index.append(idx)

    if (len(valid_index) == 0):
        obs_traj = _obs_traj
        future_traj = _future_traj
        obs_traj_e = _obs_traj_e
        future_traj_e = _future_traj_e
        feature_topview = _feature_topview
        R_map = _R_map
        R_traj = _R_traj
        num_agents = _num_agents
    else:
        num_repeat = len(_num_agents) - len(valid_index)
        for i in range(num_repeat):
            idx = random.randint(0, len(valid_index)-1)
            obs_traj.append(_obs_traj[valid_index[idx]])
            future_traj.append(_future_traj[valid_index[idx]])
            obs_traj_e.append(_obs_traj_e[valid_index[idx]])
            future_traj_e.append(_future_traj_e[valid_index[idx]])
            feature_topview.append(_feature_topview[valid_index[idx]])
            R_map.append(_R_map[valid_index[idx]])
            R_traj.append(_R_traj[valid_index[idx]])
            num_agents.append(_num_agents[valid_index[idx]])

    _len = [objs for objs in num_agents]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj_cat = torch.cat(obs_traj, dim=1)
    future_traj_cat = torch.cat(future_traj, dim=1)
    obs_traj_cat_e = torch.cat(obs_traj_e, dim=1)
    future_traj_cat_e = torch.cat(future_traj_e, dim=1)
    feature_topview_cat = torch.cat(feature_topview, dim=0)
    R_map_cat = torch.cat(R_map, dim=0)
    R_traj_cat = torch.cat(R_traj, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    return tuple([obs_traj_cat, future_traj_cat, obs_traj_cat_e, future_traj_cat_e,
                  feature_topview_cat, R_map_cat, R_traj_cat, seq_start_end, num_agents])

def seq_collate_typeF(data):


    obs_traj_ta, future_traj_ta, obs_traj_ngh, future_traj_ngh, map, neg_traj, dist2gt_cost, \
    dist2ngh_cost, drivable_cost, num_neighbors, valid_neighbor = zip(*data)

    _len = [objs for objs in num_neighbors]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj_ta_cat = torch.cat(obs_traj_ta, dim=1)
    future_traj_ta_cat = torch.cat(future_traj_ta, dim=1)
    obs_traj_ngh_cat = torch.cat(obs_traj_ngh, dim=1)
    future_traj_ngh_cat = torch.cat(future_traj_ngh, dim=1)

    neg_traj_cat = torch.cat(neg_traj, dim=2)
    dist2gt_cost_cat = torch.cat(dist2gt_cost, dim=0)
    dist2ngh_cost_cat = torch.cat(dist2ngh_cost, dim=0)
    drivable_cost_cat = torch.cat(drivable_cost, dim=0)

    map_cat = torch.cat(map, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)

    return tuple([obs_traj_ta_cat, future_traj_ta_cat, obs_traj_ngh_cat, future_traj_ngh_cat,
                  map_cat, neg_traj_cat, dist2gt_cost_cat, dist2ngh_cost_cat, drivable_cost_cat,
                  seq_start_end, np.array(valid_neighbor)])

def seq_collate_typeG(data):

    obs_traj_scene_norm, pred_traj_scene_norm, obs_vel, pred_vel, scene_center, cur_center, pred_traj = zip(*data)

    return tuple([obs_traj_scene_norm, pred_traj_scene_norm, obs_vel, pred_vel, scene_center, cur_center, pred_traj])