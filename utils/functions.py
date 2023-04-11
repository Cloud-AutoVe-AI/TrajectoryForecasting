from utils.libraries import *

def read_config():
    with open('./config/config.json', 'r') as f:
        json_data = json.load(f)
    return json_data


def get_dtypes(useGPU=True):

    if (useGPU):
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    else:
        long_dtype = torch.LongTensor
        float_dtype = torch.FloatTensor

    return long_dtype, float_dtype

def init_weights(m):

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def from_list_to_tensor_to_cuda(x, dtype):

    '''
    x : a list of (seq_len x input_dim)
    '''

    # batch_size x seq_len x input_dim
    x = np.array(x)
    if (len(x.shape) == 2):
        x = np.expand_dims(x, axis=0)

    y = torch.from_numpy(x).permute(1, 0, 2)

    return y.type(dtype).cuda()

def toNP(x):

    return x.detach().to('cpu').numpy()

def toTS(x, dtype):

    return torch.from_numpy(x).to(dtype)

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

# update, 220215
def ramp_curve(x0, x1, y0, y1, x):
    if (x < x0):
        return y0
    elif (x >= x0 and x < x1):
        return (x-x0)*(y1 - y0)/(x1 - x0) - y0
    else:
        return y1

def rotate_around_point(xy, degree, origin=(0, 0)):

    radians = math.radians(degree)
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy

def random_rotation_augmentation(traj_data, obs_len):

    seq_len, num_vids, dim = traj_data.shape
    traj_data_rt = []

    x_init = np.mean(traj_data[obs_len-1, :, 0])
    y_init = np.mean(traj_data[obs_len-1, :, 1])

    degree = random.randint(0, 359)
    for v in range(num_vids):

        x0 = np.copy(traj_data[:, v, 0]) - x_init
        y0 = np.copy(traj_data[:, v, 1]) - y_init

        xr, yr = rotate_around_point((x0, y0), degree, origin=(0, 0))
        xr += x_init
        yr += y_init

        xyr = np.concatenate([xr.reshape(seq_len, 1), yr.reshape(seq_len, 1)], axis=1)
        xyr = np.around(xyr, decimals=3)
        traj_data_rt.append(np.expand_dims(xyr, axis=1))

    if (num_vids == 1):
        return traj_data_rt[0]
    else:
        return np.concatenate(traj_data_rt, axis=1)

def save_read_latest_checkpoint_num(path, val, isSave):

    file_name = path + '/checkpoint.txt'
    index = 0

    if (isSave):
        file = open(file_name, "w")
        file.write(str(int(val)) + '\n')
        file.close()
    else:
        if (os.path.exists(file_name) == False):
            print('[Error] there is no such file in the directory')
            sys.exit()
        else:
            f = open(file_name, 'r')
            line = f.readline()
            index = int(line[:line.find('\n')])
            f.close()

    return index

def read_all_saved_param_idx(path):
    ckp_idx_list = []
    files = sorted(glob.glob(os.path.join(path, '*.pt')))
    for i, file_name in enumerate(files):
        start_idx = 0
        for j in range(-3, -10, -1):
            if (file_name[j] == '_'):
                start_idx = j+1
                break
        ckp_idx = int(file_name[start_idx:-3])
        ckp_idx_list.append(ckp_idx)
    return ckp_idx_list[::-1]

def copy_chkpt_every_N_epoch(args):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index

    root_path = args.model_dir + str(args.exp_id)
    save_directory =  root_path + '/copies'
    if save_directory != '' and not os.path.exists(save_directory):
        os.makedirs(save_directory)

    fname_list = []
    fnum_list = []
    all_file_names = os.listdir(root_path)
    for fname in all_file_names:
        if "saved" in fname:
            chk_index = get_file_number(fname)
            fname_list.append(fname)
            fnum_list.append(chk_index)

    max_idx = np.argmax(np.array(fnum_list))
    target_file = fname_list[max_idx]

    src = root_path + '/' + target_file
    dst = save_directory + '/' + target_file
    shutil.copy2(src, dst)

    print(">> {%s} is copied to {%s}" % (target_file, save_directory))

def remove_past_checkpoint(path, max_num=5):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index


    num_remain = max_num - 1
    fname_list = []
    fnum_list = []

    all_file_names = os.listdir(path)
    for fname in all_file_names:
        if "saved" in fname:
            chk_index = get_file_number(fname)
            fname_list.append(fname)
            fnum_list.append(chk_index)

    if (len(fname_list)>num_remain):
        sort_results = np.argsort(np.array(fnum_list))
        for i in range(len(fname_list)-num_remain):
            del_file_name = fname_list[sort_results[i]]
            os.remove('./' + path + '/' + del_file_name)

def print_current_train_progress(e, b, num_batchs, time_spent):

    if b == num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r [Epoch %02d] %d / %d (%.4f sec/sample)' % (e, b, num_batchs, time_spent)),

    sys.stdout.flush()

def print_current_valid_progress(b, num_batchs):

    if b >= num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r >> validation process (%d / %d) ' % (b, num_batchs)),

    sys.stdout.flush()

def print_current_test_progress(b, num_batchs):

    if b >= num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r >> test process (%d / %d) ' % (b, num_batchs)),

    sys.stdout.flush()

# update, 220209
def print_current_collision_progress(b, num_batchs, num_collisions, remaining_time):

    if b >= num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r >> [collision test progress %d / %d, remaining time %.2f hrs] num collisions : %d '
                         % (b, num_batchs, remaining_time/3600.0, num_collisions)),

    sys.stdout.flush()

def print_voxelization_progress(b, num_batchs):

    if (b >= num_batchs-1):
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r >> voxelization process (%d / %d) ' % (b, num_batchs)),

    sys.stdout.flush()

def print_training_info(args):

    try:
        model_mode = args.model_mode
    except:
        model_mode = 'vehicle'


    print("--------- %s / %s (%s) ----------" % (args.dataset_type, args.model_name, model_mode)) # update, 211005
    print(" Exp id : %d" % args.exp_id)
    print(" Gpu num : %d" % args.gpu_num)
    print(" Num epoch : %d" % args.num_epochs)
    print(" Batch size : %d" % args.batch_size)
    print(" best_k : %d" % args.best_k)
    print(" alpha/beta/gamma : %.4f/%.4f/%.4f" % (args.alpha, args.beta, args.gamma))
    print(" initial learning rate : %.5f " % args.learning_rate)
    print(" gradient clip : %.4f" % (args.grad_clip))
    print("         -----------          ")
    print(" Past horizon seconds (obs_len): %.1f sec (%d)" % (args.past_horizon_seconds, args.past_horizon_seconds*args.target_sample_period))
    print(" Future horizon seconds (pred_len) : %.1f sec (%d) "  % (args.future_horizon_seconds, args.future_horizon_seconds*args.target_sample_period))
    print(" Target sample period : %.1f Hz" % args.target_sample_period)
    print(" Traj preprocess : %d" % args.preprocess_trajectory)
    print(" Turn scene repeat number : %d" % args.num_turn_scene_repeats)
    print("         -----------          ")
    print(" Num consecutive lidar sweeps : %d" % args.num_lidar_sweeps)
    print(" Limit range : %d" % args.limit_range)
    print(" Lidar range : %d~%d" % (args.x_range_min, args.x_range_max))
    print(" Map size : %d" % args.map_size)
    print(" Load preprocessed lidar sweeps : %d" % args.use_preprocessed_lidar)
    print("         -----------          ")
    print(" LR scheduling (min lr : %.5f): %d" % (args.min_learning_rate, args.apply_lr_scheduling))
    print(" Separate LR for CNN (lr: %.5f) : %d" % (args.learning_rate_cnn, args.separate_lr_for_cnn))
    print(" Limit range change prob: %.2f" % args.limit_range_change_prob)
    print(" Stop agent remove prob : %.2f" % args.stop_agents_remove_prob)
    if (args.model_name == 'ilvm'):
        print("         -----------          ")
        print(" Cyclic scheduling for KLD (n_cycle: %d, w_ratio: %.2f) : %d" % (args.n_cycle, args.warmup_ratio, args.apply_cyclic_schedule))
        print(" Gauss Prior : %d" % args.is_gauss_prior)
        print(" Use past traj : %d" % args.use_past_traj)
        print(" Heading est method (0: traj, 1: bbox) : %d " % args.heading_est_method)

    elif (args.model_name == 'autove'):
        print("         -----------          ")
        print(" Is BBOX Heading : %d " % args.is_bbox_heading)
        print(" Traj Enc Type (0:LSTM, 1:Res+LSTM) : %d " % args.traj_enc_type)
        print(" ROI grid size / num grid  : %.2f / %d" % (args.roi_grid_size, args.roi_num_grid))
        print(" Feat Map Size  : %d " % (args.feat_map_size))
        print(" Latent Dim  : %d " % (args.latent_dim))
        print(" Dec drop prob  : %.2f " % (args.dec_drop_prob))
        print("         -----------          ")
        print(" HDmap Type (0: color, 1: multi-ch): %d" % args.hdmap_type)
        print(" L2 loss type (0:avg, 1:BOS, 2:EWTA (%.2f))  : %d" % (args.EWTA_ratio, args.l2_loss_type))
        print(" Cyclic scheduling for KLD (n_cycle: %d, w_ratio: %.2f) : %d" % (args.n_cycle, args.warmup_ratio, args.apply_cyclic_schedule))
        print(" Train w/ nuScenes : %d " % (args.is_train_w_nuscenes))
        print(" Random Flip Prob : %.2f " % args.random_flip_prob)
        print(" Agent Sampling Thr/Prob: %.2f/%.2f " % (args.agent_sampling_thr, args.agent_sampling_prob))
        print(" Pos Noise Prob: %.2f" % args.pos_noise_prob)
        print(" Add Heading Noise : %d (std: %.2f) " % (args.add_heading_noise, args.heading_noise_deg_std))
        print(" Is Shift Lane Deg : %d " % args.is_shift_lane_deg)
        print(" Is Use Proc Traj : %d " % args.is_use_proc_traj)

    elif (args.model_name == 'autove_ped'):
        print("         -----------          ")
        print(" Train w/ nuScenes : %d " % (args.is_train_w_nuscenes))
        print(" Train Dis. : %d " % (args.is_train_dis))
        print(" L2 loss type (0:avg, 1:BOS, 2:EWTA (%.2f))  : %d" % (args.EWTA_ratio, args.l2_loss_type))
        print(" Random Rot. Augmentation  : %d " % (args.is_random_rotate))
        print(" Cyclic scheduling for KLD (n_cycle: %d, w_ratio: %.2f) : %d" % (
        args.n_cycle, args.warmup_ratio, args.apply_cyclic_schedule))
        print(" ROI grid size / num grid  : %.2f / %d" % (args.roi_grid_size, args.roi_num_grid))
        print(" Feat Map Size  : %d " % (args.feat_map_size))
        print(" Latent Dim  : %d " % (args.latent_dim))
        print(" Dec drop prob  : %.2f " % (args.dec_drop_prob))
    elif (args.model_name == 'scratch_t3'):
        print("         -----------          ")
        print(" Repeat Scene : %d" % args.is_repeat_scene)
        print(" Num possible paths : %d" % args.num_max_paths)
        print(" Random path order : %d" % args.is_random_path_order)
        print(" Social Pooling Type : %d" % args.social_pooling_type)
        print(" Train Dis (GAN prior prob. %.2f) : %d" % (args.gan_prior_prob, args.is_train_dis))
        print(" Forward path length (m) : %.1f" % args.max_path_len_forward)
        print(" Path resolution (m) : %.1f" % args.path_resol)
        print(" Cyclic scheduling for KLD (n_cycle: %d, w_ratio: %.2f) : %d" % (
        args.n_cycle, args.warmup_ratio, args.apply_cyclic_schedule))
    elif (args.model_name == 'baseline'):
        print("         -----------          ")
        print(" BOS Objective : %d" % args.is_bos_obj)
        print(" Repeat Scene : %d" % args.is_repeat_scene)
        print(" Num possible paths : %d" % args.num_max_paths)
        print(" Random path order : %d" % args.is_random_path_order)
        print(" Forward path length (m) : %.1f" % args.max_path_len_forward)
        print(" Path resolution (m) : %.1f" % args.path_resol)
        print(" Cyclic scheduling for KLD (n_cycle: %d, w_ratio: %.2f) : %d" % (
        args.n_cycle, args.warmup_ratio, args.apply_cyclic_schedule))
    elif (args.model_name == 'baseline_nf'):
        print("         -----------          ")
        print(" BOS Objective : %d" % args.is_bos_obj)
        print(" Repeat Scene : %d" % args.is_repeat_scene)
        print(" Num possible paths : %d" % args.num_max_paths)
        print(" Random path order : %d" % args.is_random_path_order)
        print(" Forward path length (m) : %.1f" % args.max_path_len_forward)
        print(" Path resolution (m) : %.1f" % args.path_resol)
        print(" Cyclic scheduling for KLD (n_cycle: %d, w_ratio: %.2f) : %d" % (
        args.n_cycle, args.warmup_ratio, args.apply_cyclic_schedule))
    elif (args.model_name == 'NMP'):
        print("         -----------          ")
        print(" Collision Distance : %.2f" % args.collision_dist)
        print(" Eval. Method (0: minADE, 1: minCost) : %d" % args.eval_method)
    print("----------------------------------")
