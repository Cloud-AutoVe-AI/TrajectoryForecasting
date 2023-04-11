from utils.functions import *
from utils.loss import *
from Onnx.onnx_utils import *

class Solver:

    def __init__(self, args, num_train_scenes, dtype):

        # training setting
        self.args = args
        self.dtype = dtype
        self.best_k = args.best_k
        self.beta = args.beta
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.grad_clip = args.grad_clip
        self.n_cycle = args.n_cycle
        self.l2_loss_type = args.l2_loss_type # update, 220107
        self.EWTA_ratio = args.EWTA_ratio # update, 220107
        self.warmup_ratio = args.warmup_ratio
        self.num_batches = int(num_train_scenes / args.batch_size)
        self.total_num_iteration = args.num_epochs * self.num_batches
        folder_name = args.dataset_type + '_' + args.model_name + '_model' + str(args.exp_id)
        self.save_dir = os.path.join('./saved_models/', folder_name)

        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)

        # training monitoring
        self.iter = 0
        self.l2_losses = 0
        self.kld_losses = 0
        self.prev_ADE = 1e5
        self.init_lr = args.learning_rate
        self.cur_lr = args.learning_rate
        self.min_lr = args.min_learning_rate

        # define models
        from models.autove_ped import ConvNet, Scratch
        self.cnn = ConvNet(args=args)
        if (args.multi_gpu == 1):
            self.cnn = nn.DataParallel(self.cnn)
            self.cnn.type(dtype)

        self.scratch = Scratch(args=args)
        self.scratch.type(dtype)

        # define optimizer
        if (args.separate_lr_for_cnn == 1):
            self.opt = optim.Adam([{'params': self.scratch.parameters()},
                                   {'params': self.cnn.parameters(), 'lr': args.learning_rate_cnn}],
                                  lr=args.learning_rate)
        else:
            self.opt = optim.Adam(list(self.cnn.parameters()) + list(self.scratch.parameters()), lr=args.learning_rate)



        # training scheduling
        self.apply_kld_scheduling = args.apply_cyclic_schedule
        self.apply_lr_scheduling = args.apply_lr_scheduling

        self.kld_weight_scheduler = self.create_kld_weight_scheduler()
        self.EWTA_scheduler = self.create_EWTA_scheduler() # update, 220107
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.9999)

        # load network params
        if (args.load_pretrained == 1):
            ckp_idx = save_read_latest_checkpoint_num(os.path.join(self.save_dir), 0, isSave=False)
            _ = self.load_pretrained_network_params(ckp_idx)

        print(">> Optimizer is loaded from {%s} " % os.path.basename(__file__))


    def mode_selection(self, isTrain=True):
        if (isTrain):
            self.cnn.train()
            self.scratch.train()
        else:
            self.cnn.eval()
            self.scratch.eval()

    def init_loss_tracker(self):
        self.l2_losses = 0
        self.kld_losses = 0

    def normalize_loss_tracker(self):
        self.l2_losses /= self.num_batches
        self.kld_losses /= self.num_batches

    def create_kld_weight_scheduler(self):
        scheduler = frange_cycle_linear(self.total_num_iteration, n_cycle=self.n_cycle, ratio=self.warmup_ratio)

        if (self.apply_kld_scheduling == 1):
            return scheduler
        else:
            return np.ones_like(scheduler)

    # update, 220107
    def create_EWTA_scheduler(self):
        # scheduler = -1 * frange_cycle_linear(2000 * 100, n_cycle=1, ratio=EWTA_ratio) + 1
        scheduler = -1 * frange_cycle_linear(self.total_num_iteration, n_cycle=1, ratio=self.EWTA_ratio) + 1
        scheduler = ((self.best_k - 1) * scheduler + 1).astype('int')

        chk = scheduler > self.best_k
        scheduler[chk] = self.best_k

        chk = scheduler < 1
        scheduler[chk] = 1

        # plt.plot(scheduler.astype('float'))
        # plt.show()

        return scheduler


    def learning_rate_step(self, e):

        if (e >= 30 and e < 60):
            for g in self.opt.param_groups:
                g['lr'] = self.init_lr * 0.1
        elif (e >= 60):
            for g in self.opt.param_groups:
                g['lr'] = self.init_lr * 0.01

        self.cur_lr = self.opt.param_groups[0]['lr']

    def load_pretrained_network_params(self, ckp_idx):

        file_name = self.save_dir + '/saved_chk_point_%d.pt' % ckp_idx
        checkpoint = torch.load(file_name)
        self.cnn.load_state_dict(checkpoint['cnn_state_dict'])
        self.scratch.load_state_dict(checkpoint['scratch_state_dict'])
        self.opt.load_state_dict(checkpoint['opt'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.iter = checkpoint['iter']
        self.prev_ADE = checkpoint['ADE']
        print('>> trained parameters are loaded from {%s}' % file_name)
        print(">> current training settings ...")
        print("   . iteration : %.d" % (self.iter))
        print("   . prev_ADE : %.4f" % (self.prev_ADE))

        return ckp_idx

    def save_trained_network_params(self, e):

        # save trained model
        _ = save_read_latest_checkpoint_num(os.path.join(self.save_dir), e, isSave=True)
        file_name = self.save_dir + '/saved_chk_point_%d.pt' % e
        check_point = {
            'epoch': e,
            'cnn_state_dict': self.cnn.state_dict(),
            'scratch_state_dict': self.scratch.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'opt': self.opt.state_dict(),
            'ADE': self.prev_ADE,
            'iter': self.iter}
        torch.save(check_point, file_name)
        print(">> current network is saved ...")
        remove_past_checkpoint(os.path.join('./', self.save_dir), max_num=self.args.max_num_chkpts)

    def print_status(self, e, time_left):

        # update, 220107
        if (self.iter > len(self.kld_weight_scheduler)-1):
            beta_scheduler = self.kld_weight_scheduler[-1]
        else:
            beta_scheduler = self.kld_weight_scheduler[self.iter]

        if (self.iter > len(self.EWTA_scheduler) - 1):
            EWTA_scheduler = self.EWTA_scheduler[-1]
        else:
            EWTA_scheduler = self.EWTA_scheduler[self.iter]

        print("[E %02d, %.2f hrs left] l2-loss : %.4f, kld-loss : %.4f (beta_scheduler : %.4f, current_lr : %.8f)"
              % (e, time_left, self.l2_losses, self.kld_losses, self.kld_weight_scheduler[self.iter], self.cur_lr))

    # update, 220107
    def L2_sample_average(self, pred_trajs_ego, future_traj_a):
        l2 = torch.zeros(1).to(future_traj_a.cuda())
        for i in range(self.best_k):
            l2 += l2_loss(pred_trajs_ego[i].permute(1, 0, 2), future_traj_a[:, :, :2].cuda())
        l2 = l2 / float(self.best_k)
        return l2

    # update, 220107
    def L2_Best_of_Samples(self, pred_trajs_ego, future_traj_a):
        losses_tensor, losses_np = [], []
        for i in range(self.best_k):
            losses_tensor.append(l2_loss(pred_trajs_ego[i].permute(1, 0, 2), future_traj_a[:, :, :2].cuda()))
            losses_np.append(losses_tensor[i].item())

        min_idx = np.argmin(np.array(losses_np))
        l2 = losses_tensor[min_idx]
        return l2

    # update, 220107
    def L2_EWTA(self, pred_trajs_ego, future_traj_a):

        if (self.iter > len(self.EWTA_scheduler)-1):
            K = self.EWTA_scheduler[-1]
        else:
            K = self.EWTA_scheduler[self.iter]

        l2 = torch.zeros(1).to(future_traj_a.cuda())
        losses_tensor, losses_np = [], []
        for i in range(self.best_k):
            losses_tensor.append(l2_loss(pred_trajs_ego[i].permute(1, 0, 2), future_traj_a[:, :, :2].cuda()))
            losses_np.append(losses_tensor[i].item())

        sort_idx = np.argsort(np.array(losses_np))
        for k in range(K):
            l2 += losses_tensor[sort_idx[k]]
        l2 = l2 / float(K)

        return l2

    # update, 220107
    def generation_step(self, obs_traj, future_traj, obs_traj_a, future_traj_a, feature_topview, R_map, R_traj,
                        seq_start_end, num_agents, epoch=None):

        # process map data
        feat_map = self.cnn(feature_topview.cuda())

        # predict future trajectory
        pred_trajs_ego, m1, lv1, m2, lv2 = self.scratch(obs_traj[:, :, :2].cuda(),
                                               future_traj[:, :, :2].cuda(),
                                               obs_traj_a[:, :, :2].cuda(),
                                               future_traj_a[:, :, :2].cuda(),
                                               seq_start_end,
                                               feat_map,
                                               R_map.cuda(),
                                               R_traj.cuda(),
                                               self.best_k)

        # l2 loss
        if (self.l2_loss_type == 0):
            l2 = self.L2_sample_average(pred_trajs_ego, future_traj_a)
        elif (self.l2_loss_type == 1):
            l2 = self.L2_Best_of_Samples(pred_trajs_ego, future_traj_a)
        elif (self.l2_loss_type == 2):
            l2 = self.L2_EWTA(pred_trajs_ego, future_traj_a)
        self.l2_losses += l2.item()

        # kld loss
        kld = kld_loss(m1, lv1, m2, lv2)
        self.kld_losses += kld.item()

        if (self.iter > len(self.kld_weight_scheduler) - 1):
            self.iter = len(self.kld_weight_scheduler) - 1
        loss = l2 + (self.beta * self.kld_weight_scheduler[self.iter]) * kld

        # back-propagation
        self.opt.zero_grad()
        loss.backward()
        if self.grad_clip > 0.0:
            nn.utils.clip_grad_value_(self.scratch.parameters(), self.grad_clip)
            nn.utils.clip_grad_value_(self.cnn.parameters(), self.grad_clip)
        self.opt.step()

        if (self.apply_lr_scheduling == 1):
            self.learning_rate_step(epoch)

        # increase iteration number
        self.iter += 1

    # ------------------------
    # Training
    # ------------------------
    def train(self, batch, epoch=None):

        # read batch data
        obs_traj, future_traj, obs_traj_a, future_traj_a, feature_topview, \
        R_map, R_traj, seq_start_end, num_agents = batch

        # update, 220107
        if (torch.count_nonzero(torch.isnan(future_traj)) == 0):
            self.generation_step(obs_traj, future_traj, obs_traj_a, future_traj_a, feature_topview, R_map, R_traj,
                                 seq_start_end, num_agents)


    # ------------------------
    # Validation
    # ------------------------
    def eval(self, data_loader, e):

        # set to evaluation mode
        self.mode_selection(isTrain=False)

        ADE = []
        for b in range(0, len(data_loader.valid_data), self.args.valid_step):

            # data preparation
            obs_traj, future_traj, obs_traj_a, future_traj_a, map_data, R_map, R_traj, num_agents, _, _ = \
                data_loader.next_sample(b, mode='valid')

            # update, 220107
            if (np.count_nonzero(np.isnan(future_traj)) > 0):
                continue

            obs_traj_cuda = torch.from_numpy(obs_traj).type(self.dtype)
            obs_traj_a_cuda = torch.from_numpy(obs_traj_a).type(self.dtype)
            map_data_cuda = torch.from_numpy(map_data).permute(2, 0, 1).type(self.dtype)
            map_data_cuda = torch.unsqueeze(map_data_cuda, dim=0)
            seq_start_end = torch.from_numpy(np.array([[0, num_agents]]))
            m_R_cuda = torch.from_numpy(R_map).type(self.dtype)
            t_R_cuda = torch.from_numpy(R_traj).type(self.dtype)

            # process map data
            feat_map = self.cnn(map_data_cuda)

            # predict future trajectory
            pred_trajs = self.scratch.inference(obs_traj_cuda[:, :, :2],
                                       obs_traj_a_cuda[:, :, :2],
                                       seq_start_end,
                                       feat_map,
                                       m_R_cuda,
                                       t_R_cuda,
                                       self.best_k)

            seq_len, batch, _ = pred_trajs[0].size()

            for k in range(self.best_k):
                err = np.sqrt(np.sum((toNP(pred_trajs[k]) - future_traj[:, :, :2])**2, axis=2)).reshape(seq_len * batch)
                ADE += err.tolist()


            print_current_valid_progress(b, len(data_loader.valid_data))
        print(">> evaluation results are created .. {ADE:%.4f}" % np.mean(ADE))


        if (self.prev_ADE > np.mean(ADE)):
            self.prev_ADE = np.mean(ADE)
            self.save_trained_network_params(e)

    # ------------------------
    # Testing
    # ------------------------

    def padd(self, tensor, num_padd):

        seq_len, _, dim = tensor.size()
        padd = torch.zeros(size=(seq_len, num_padd, dim)).to(tensor)
        return torch.cat((tensor, padd), dim=1)


    def test(self, data, dtype, best_k):

        obs_traj, future_traj, obs_traj_a, future_traj_a, map_data, \
        R_map, R_traj, num_agents, agent_ids, scene = data

        # update, 220107
        if (np.count_nonzero(np.isnan(future_traj)) > 0):
            return [], [], [], [], [], False

        obs_traj_cuda = torch.from_numpy(obs_traj).type(self.dtype)
        obs_traj_a_cuda = torch.from_numpy(obs_traj_a).type(self.dtype)
        map_data_cuda = torch.from_numpy(map_data).permute(2, 0, 1).type(self.dtype)
        map_data_cuda = torch.unsqueeze(map_data_cuda, dim=0)
        seq_start_end = torch.from_numpy(np.array([[0, num_agents]]))
        Rm_cuda = torch.from_numpy(R_map).type(self.dtype)
        Rt_cuda = torch.from_numpy(R_traj).type(self.dtype)

        # process map data
        feat_map = self.cnn(map_data_cuda)

        # predict future trajectory
        pred_trajs_ = self.scratch.inference(obs_traj_cuda[:, :, :2],
                                            obs_traj_a_cuda[:, :, :2],
                                            seq_start_end,
                                            feat_map,
                                            Rm_cuda,
                                            Rt_cuda,
                                            self.best_k)

        # consider only valid trajs
        pred_trajs = []
        for k in range(self.best_k):
            pred_trajs.append(torch.unsqueeze(pred_trajs_[k], dim=0))
        pred_trajs = toNP(torch.cat(pred_trajs, dim=0))

        return obs_traj, future_traj, pred_trajs, agent_ids[0], scene, True



    # ------------------------
    # Convert to onnx file
    # ------------------------

    def convert_to_onnx_file_CNN(self, args, onnx_save_dir):

        # model path
        file_name = self.save_dir + '/saved_chk_point_%d.pt' % args.model_num
        checkpoint = torch.load(file_name)

        # load model
        from models.autove_onnx import ConvNet
        cnn = ConvNet(args=self.args)
        cnn.cuda().eval()
        if (self.args.multi_gpu == 1):
            self.cnn.load_state_dict(checkpoint['cnn_state_dict'])
            cnn.load_state_dict(self.cnn.module.state_dict())
        else:
            cnn.load_state_dict(checkpoint['cnn_state_dict'])

        # CNN input spec
        map_size = self.args.map_size
        ch_dim = self.args.hdmap_ch_dim

        input_names = ['input']
        output_names = ['output']

        # convert to onnx
        file_name = 'cnn_e%dm%d_ped.onnx' % (args.exp_id, args.model_num) # update, 220107
        dummy_input = torch.randn(1, ch_dim, map_size, map_size).cuda()
        torch.onnx.export(cnn, dummy_input, os.path.join(onnx_save_dir, file_name), verbose=False, input_names=input_names,
                          output_names=output_names, opset_version=11)

    def convert_to_onnx_file_TrajDec(self, args, onnx_save_dir):

        # model path
        file_name = self.save_dir + '/saved_chk_point_%d.pt' % args.model_num
        checkpoint = torch.load(file_name)

        # load model
        from models.autove_onnx import Scratch
        # important ---
        self.args.best_k = args.best_k
        self.args.max_num_agents = args.max_num_agents
        # important ---
        self.scratch_onnx = Scratch(args=self.args, onnx=True, best_k=args.best_k)
        self.scratch_onnx.type(self.dtype)
        self.scratch_onnx.eval()
        self.scratch_onnx.load_state_dict(checkpoint['scratch_state_dict'])

        # input spec
        obs_len = int(self.args.past_horizon_seconds * self.args.target_sample_period)
        max_num_agents = self.args.max_num_agents
        latent_dim = self.args.latent_dim
        best_k = self.args.best_k

        input0 = torch.randn(obs_len, max_num_agents, 2).cuda() # obs_traj
        input1 = torch.randn(obs_len, max_num_agents, 2).cuda() # obs_traj_agent
        input2 = torch.randn(max_num_agents, self.args.cnn_outch_dim, 40, 40).cuda()
        input3 = torch.randn(max_num_agents, 2, 2).cuda()
        input4 = torch.randn(best_k*max_num_agents, latent_dim).cuda()

        input_names = ['input0', 'input1', 'input2', 'input3', 'input4']
        output_names = ['output']

        # convert to onnx
        file_name = 'trajdec_e%dm%d_k%d_ped.onnx' % (args.exp_id, args.model_num, args.best_k) # update, 220107

        dummy_input = (input0, input1, input2, input3, input4)
        torch.onnx.export(self.scratch_onnx, dummy_input, os.path.join(onnx_save_dir, file_name), verbose=False,
                          input_names=input_names, output_names=output_names, opset_version=11)

    def onnx_load_splited_modules(self, ckp_idx):
        from models.autove_onnx import ConvNet, Scratch, RROI_Pooling

        # define models
        self.cnn_onnx = ConvNet(args=self.args)
        if (self.args.multi_gpu == 1):
            self.cnn_onnx = nn.DataParallel(self.cnn_onnx)
            self.cnn_onnx.type(self.dtype)
        self.cnn_onnx.eval()

        self.scratch_onnx = Scratch(args=self.args)
        self.scratch_onnx.type(self.dtype)
        self.scratch_onnx.eval()

        self.pooling = RROI_Pooling(args=self.args)

        # load model params
        file_name = self.save_dir + '/saved_chk_point_%d.pt' % ckp_idx
        checkpoint = torch.load(file_name)
        self.cnn_onnx.load_state_dict(checkpoint['cnn_state_dict'])
        self.scratch_onnx.load_state_dict(checkpoint['scratch_state_dict'])
        print('>> trained parameters are loaded from {%s}' % file_name)
        print(">> current training settings ...")
        print("   . iteration : %.d" % (self.iter))
        print("   . prev_ADE : %.4f" % (self.prev_ADE))


    def onnx_test_splited_modules(self, data):

        obs_traj, future_traj, obs_traj_a, future_traj_a, map_data, \
        R_map, R_traj, num_agents, agent_ids, scene = data

        # update, 220107
        if (np.count_nonzero(np.isnan(future_traj)) > 0):
            return [], [], [], [], [], False

        obs_traj_cuda = torch.from_numpy(obs_traj).type(self.dtype)
        obs_traj_a_cuda = torch.from_numpy(obs_traj_a).type(self.dtype)
        map_data_cuda = torch.from_numpy(map_data).permute(2, 0, 1).type(self.dtype)
        map_data_cuda = torch.unsqueeze(map_data_cuda, dim=0)
        seq_start_end = torch.from_numpy(np.array([[0, num_agents]]))
        Rm_cuda = torch.from_numpy(R_map).type(self.dtype)
        Rt_cuda = torch.from_numpy(R_traj).type(self.dtype)

        # predict future trajectory
        feat_map = self.cnn_onnx(map_data_cuda)
        agent_feat_pool = self.pooling(obs_traj_cuda, feat_map, Rm_cuda, seq_start_end)
        pred_trajs_onnx, _, _ = self.scratch_onnx.inference(obs_traj_cuda[:, :, :2],
                                            obs_traj_a_cuda[:, :, :2],
                                            agent_feat_pool,
                                            Rm_cuda,
                                            self.best_k)

        pred_trajs_onnx = toNP(pred_trajs_onnx.reshape(self.pred_len, self.best_k, num_agents, 2).permute(1, 0, 2, 3))

        return obs_traj, future_traj, pred_trajs_onnx, agent_ids, scene, True

    def read_onnx_files(self, args, onnx_save_dir):

        file_name_cnn = 'cnn_e%dm%d_ped.onnx' % (args.exp_id, args.model_num)  # update, 220107
        file_name_trajdec = 'trajdec_e%dm%d_k%d_ped.onnx' % (args.exp_id, args.model_num, args.best_k)  # update, 220107

        sess = onnxruntime.InferenceSession(os.path.join(onnx_save_dir, file_name_cnn), None)
        sess0 = dict({'sess' : sess,
                     'input0' : sess.get_inputs()[0].name,
                     'output0': sess.get_outputs()[0].name})

        sess = onnxruntime.InferenceSession(os.path.join(onnx_save_dir, file_name_trajdec), None)
        sess1 = dict({'sess' : sess,
                     'input0' : sess.get_inputs()[0].name,
                      'input1': sess.get_inputs()[1].name,
                      'input2': sess.get_inputs()[2].name,
                      'output0': sess.get_outputs()[0].name})


        self.onnx_models = [sess0, sess1]

    def onnx_compare_pytorch_onnx_models(self, data, max_num_agents):

        obs_traj, future_traj, obs_traj_a, future_traj_a, map_data,\
        R_map, R_traj, num_agents, agent_ids, scene = data

        # update, 220107
        if (np.count_nonzero(np.isnan(future_traj)) > 0):
            return -1

        if (num_agents > max_num_agents):
            return -1

        # -------------------------
        # Pytorch Model
        # -------------------------
        obs_traj_cuda = torch.from_numpy(obs_traj).type(self.dtype)
        obs_traj_a_cuda = torch.from_numpy(obs_traj_a).type(self.dtype)
        map_data_cuda = torch.from_numpy(map_data).permute(2, 0, 1).type(self.dtype)
        map_data_cuda = torch.unsqueeze(map_data_cuda, dim=0)
        seq_start_end = torch.from_numpy(np.array([[0, num_agents]]))
        Rm_cuda = torch.from_numpy(R_map).type(self.dtype)
        Rt_cuda = torch.from_numpy(R_traj).type(self.dtype)

        # process map data
        feat_map = self.cnn_onnx(map_data_cuda)
        agent_feat_pool = self.pooling(obs_traj_cuda, feat_map, Rm_cuda, seq_start_end)

        # predict future trajectory, (best_k x batch) x seq_len x 2
        _, pred_trajs, Zs = self.scratch_onnx.inference(obs_traj_cuda[:, :, :2],
                                            obs_traj_a_cuda[:, :, :2],
                                            agent_feat_pool,
                                            Rt_cuda,
                                            self.best_k)




        # -------------------------
        # Onnx Model
        # -------------------------
        # process map data
        feat_map_onnx = self.onnx_models[0]['sess'].run([self.onnx_models[0]['output0']], {self.onnx_models[0]['input0']: toNP(map_data_cuda)})[0]
        Errs = np.abs(toNP(feat_map) - feat_map_onnx)
        print(">> CNN Err Max : %.5f" % np.max(Errs))



        # reshape data for onnx inputs -----------------------
        num_valid_agents = obs_traj_cuda.size(1)

        # obs_traj_onnx_cuda = torch.zeros(size=(obs_traj_cuda.size(0), max_num_agents, 2)).to(obs_traj_cuda)
        # obs_traj_onnx_cuda[:, :num_valid_agents, :2] = obs_traj_cuda[:, :, :2]

        obs_traja_onnx_cuda = torch.zeros(size=(obs_traj_cuda.size(0), max_num_agents, 2)).to(obs_traj_cuda)
        obs_traja_onnx_cuda[:, :num_valid_agents, :2] = obs_traj_a_cuda[:, :, :2]

        actor_features_onnx = torch.zeros(size=(max_num_agents, agent_feat_pool.size(1), agent_feat_pool.size(2), agent_feat_pool.size(3))).to(agent_feat_pool)
        actor_features_onnx[:num_valid_agents] = agent_feat_pool

        # Rm_onnx_cuda = torch.zeros(size=(max_num_agents, 2, 2)).to(Rm_cuda)
        # Rm_onnx_cuda[:num_valid_agents] = Rm_cuda

        # Rt_onnx_cuda = torch.zeros(size=(max_num_agents, 2, 2)).to(Rt_cuda)
        # Rt_onnx_cuda[:num_valid_agents] = Rt_cuda

        Z_onnx_cuda = torch.zeros(size=(max_num_agents*self.best_k, self.args.latent_dim)).to(obs_traj_cuda)
        for k in range(self.best_k):
            Zsk = Zs[k*num_valid_agents:(k+1)*num_valid_agents]
            Z_onnx_cuda[k*max_num_agents:k*max_num_agents + num_valid_agents] = Zsk

        # final
        pred_trajs_onnx = self.onnx_models[1]['sess'].run([self.onnx_models[1]['output0']],
                                                         {self.onnx_models[1]['input0']: toNP(obs_traja_onnx_cuda),
                                                          self.onnx_models[1]['input1']: toNP(actor_features_onnx),
                                                          self.onnx_models[1]['input2']: toNP(Z_onnx_cuda)})[0]



        Errs = []
        for k in range(self.best_k):
            output0k = toNP(pred_trajs[k*num_valid_agents:(k+1)*num_valid_agents])
            output1k = pred_trajs_onnx[k*max_num_agents:(k+1)*max_num_agents][:num_valid_agents]
            max_err = np.max(np.abs(output0k - output1k))
            Errs.append(max_err)
        print(">> Pred Traj Err Max : %.5f" % np.max(Errs))

        return 1

    def onnx_make_bin_samples(self, data, max_num_agents, save_dir, b):



        obs_traj, future_traj, obs_traj_a, future_traj_a, map_data, \
        R_map, R_traj, num_agents, agent_ids, scene = data

        # update, 220107
        if (np.count_nonzero(np.isnan(future_traj)) > 0):
            return -1

        if (num_agents > max_num_agents):
            return -1

        obs_traj_cuda = torch.from_numpy(obs_traj).type(self.dtype)
        # note : need to remove nan
        chk = torch.isnan(obs_traj_cuda)
        obs_traj_cuda[chk] = 0.0
        obs_traj_a_cuda = torch.from_numpy(obs_traj_a).type(self.dtype)
        map_data_cuda = torch.from_numpy(map_data).permute(2, 0, 1).type(self.dtype)
        map_data_cuda = torch.unsqueeze(map_data_cuda, dim=0)
        seq_start_end = torch.from_numpy(np.array([[0, num_agents]]))
        Rm_cuda = torch.from_numpy(R_map).type(self.dtype)
        Rt_cuda = torch.from_numpy(R_traj).type(self.dtype)


        '''Map data'''
        file_name = 'map_%04d.bin' % b
        save_as_bin_file_4d(toNP(map_data_cuda), os.path.join(save_dir, file_name), 'd')

        # -------------------------
        # Pytorch Model
        # -------------------------
        feat_map = self.cnn_onnx(map_data_cuda)

        '''Feat_map data'''
        file_name = 'feat_%04d.bin' % b
        save_as_bin_file_4d(toNP(feat_map), os.path.join(save_dir, file_name), 'd')

        '''Actor feature data'''
        agent_feat_pool = self.pooling(obs_traj_cuda, feat_map, Rm_cuda, seq_start_end)
        actor_features_onnx = torch.zeros(size=(max_num_agents, agent_feat_pool.size(1), agent_feat_pool.size(2), agent_feat_pool.size(3))).to(agent_feat_pool)
        actor_features_onnx[:num_agents] = agent_feat_pool
        file_name = 'actorfeat_%04d.bin' % b
        save_as_bin_file_4d(toNP(actor_features_onnx), os.path.join(save_dir, file_name), 'd')



        # -------------------------
        # Onnx Model
        # -------------------------
        ''' N '''
        num_agents_np = num_agents * np.ones(shape=(1, 1))
        file_name = 'N_%04d.bin' % b
        save_as_bin_file_2d(num_agents_np, os.path.join(save_dir, file_name), 'd')

        # ''' Ht & Hb'''
        # file_name = 'Ht_%04d.bin' % b
        # save_as_bin_file_2d(Ht.reshape(1, num_agents), os.path.join(save_dir, file_name), 'd')
        #
        # file_name = 'Hb_%04d.bin' % b
        # save_as_bin_file_2d(Hb.reshape(1, num_agents), os.path.join(save_dir, file_name), 'd')


        ''' Obs Traj '''
        obs_traj_onnx_cuda = torch.zeros(size=(obs_traj_cuda.size(0), max_num_agents, 2)).to(obs_traj_cuda)
        obs_traj_onnx_cuda[:, :num_agents, :2] = obs_traj_cuda[:, :, :2]
        file_name = 'obstraj_onnx_%04d.bin' % b
        save_as_bin_file_3d(toNP(obs_traj_onnx_cuda[:, :, :2]), os.path.join(save_dir, file_name), 'd')

        aa = read_bin_file_3d(os.path.join(save_dir, file_name), 'd', (obs_traj_cuda.size(0), max_num_agents, 2))


        ''' Obs Traj Agent Centric'''
        obs_traja_onnx_cuda = torch.zeros(size=(obs_traj_cuda.size(0), max_num_agents, 2)).to(obs_traj_cuda)
        obs_traja_onnx_cuda[:, :num_agents, :2] = obs_traj_a_cuda[:, :, :2]
        file_name = 'obstraja_onnx_%04d.bin' % b
        save_as_bin_file_3d(toNP(obs_traja_onnx_cuda[:, :, :2]), os.path.join(save_dir, file_name), 'd')

        ''' Rm '''
        Rm_onnx_cuda = torch.zeros(size=(max_num_agents, 2, 2)).to(Rm_cuda)
        Rm_onnx_cuda[:num_agents] = Rm_cuda
        file_name = 'Rm_onnx_%04d.bin' % b
        save_as_bin_file_3d(toNP(Rm_onnx_cuda), os.path.join(save_dir, file_name), 'd')

        ''' Rt '''
        Rt_onnx_cuda = torch.zeros(size=(max_num_agents, 2, 2)).to(Rt_cuda)
        Rt_onnx_cuda[:num_agents] = Rt_cuda
        file_name = 'Rt_onnx_%04d.bin' % b
        save_as_bin_file_3d(toNP(Rt_onnx_cuda), os.path.join(save_dir, file_name), 'd')

        ''' Z '''
        Z = np.around(np.random.randn(max_num_agents*self.best_k, self.args.latent_dim), 4)
        Z_onnx_cuda = torch.from_numpy(Z).to(obs_traj_cuda)
        file_name = 'Z_%04d.bin' % b
        save_as_bin_file_2d(toNP(Z_onnx_cuda), os.path.join(save_dir, file_name), 'd')


        pred_trajs_onnx = self.onnx_models[1]['sess'].run([self.onnx_models[1]['output0']],
                                                         {self.onnx_models[1]['input0']: toNP(obs_traja_onnx_cuda),
                                                          self.onnx_models[1]['input1']: toNP(actor_features_onnx),
                                                          self.onnx_models[1]['input2']: toNP(Z_onnx_cuda)})[0]
        '''pred traj '''
        file_name = 'predtraj_onnx_%04d.bin' % b
        save_as_bin_file_3d(pred_trajs_onnx, os.path.join(save_dir, file_name), 'd')
