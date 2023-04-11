from models.base_modules import *
from torchvision import models

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

class Header(nn.Module):

    def __init__(self, in_ch_dim, out_ch_dim, use_bn=True):
        super(Header, self).__init__()

        self.use_bn = use_bn
        bias = not use_bn
        self.conv1 = conv3x3(in_ch_dim, out_ch_dim, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_ch_dim)
        self.conv2 = conv3x3(out_ch_dim, out_ch_dim, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_ch_dim)
        self.conv3 = conv3x3(out_ch_dim, out_ch_dim, bias=bias)
        self.bn3 = nn.BatchNorm2d(out_ch_dim)
        self.conv4 = conv3x3(out_ch_dim, out_ch_dim, bias=bias)
        self.bn4 = nn.BatchNorm2d(out_ch_dim)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)

        return x

class ConvNet(nn.Module):

    def __init__(self, args):
        super(ConvNet, self).__init__()

        use_pretrained = True
        # self.conv0 = list(models.resnet50(pretrained=use_pretrained).children())[0]
        self.conv0 = nn.Conv2d(args.hdmap_ch_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.conv1 = list(models.resnet50(pretrained=use_pretrained).children())[1]
        self.conv2 = list(models.resnet50(pretrained=use_pretrained).children())[2]
        self.conv3 = list(models.resnet50(pretrained=use_pretrained).children())[3]
        self.conv4 = list(models.resnet50(pretrained=use_pretrained).children())[4]
        self.conv5 = list(models.resnet50(pretrained=use_pretrained).children())[5]

        self.set_parameter_requires_grad(self.conv0)
        self.set_parameter_requires_grad(self.conv1)
        self.set_parameter_requires_grad(self.conv2)
        self.set_parameter_requires_grad(self.conv3)
        self.set_parameter_requires_grad(self.conv4)
        self.set_parameter_requires_grad(self.conv5)

        self.maxpoolx2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.header = Header(in_ch_dim=640, out_ch_dim=args.cnn_outch_dim)


    def set_parameter_requires_grad(self, model, feature_extracting=False):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def _upsample_add(self, x, H, W):
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

    def forward(self, x):

        x0 = self.conv0(x)   # x 1/2
        x1 = self.conv1(x0)  # x 1
        x2 = self.conv2(x1)  # x 1
        x3 = self.conv3(x2)  # x 1/2
        x4 = self.conv4(x3)  # x 1
        x5 = self.conv5(x4)  # x 1/2

        # multi-resolution concat
        x3_0 = self.maxpoolx2(x0)
        x3_5 = self._upsample_add(x5, x3.size(2), x3.size(3))
        cat = torch.cat((x3_0, x3, x3_5), dim=1)

        return self.header(cat)


class RROI_Pooling(nn.Module):

    def __init__(self, args):
        super(RROI_Pooling, self).__init__()

        # agent's related params
        self.model_mode = args.model_mode # update, 220106
        self.max_num_agents = args.max_num_agents
        self.agent_mapfeat_dim = args.agent_mapfeat_dim

        # map related params
        self.map_size = args.feat_map_size
        axis_range_y = args.y_range_max - args.y_range_min
        axis_range_x = args.x_range_max - args.x_range_min
        self.scale_y = float(self.map_size - 1) / axis_range_y
        self.scale_x = float(self.map_size - 1) / axis_range_x
        self.y_range_max = args.y_range_max
        self.x_range_max = args.x_range_max

        self.roi_grid_size = args.roi_grid_size
        self.roi_num_grid = args.roi_num_grid

    def clip(self, array):

        OOB = np.logical_or((array < 0), (array > self.map_size - 1))

        array[array < 0] = 0
        array[array > self.map_size - 1] = self.map_size - 1

        return array, OOB

    def create_ROI_template(self, grid_size, num_grid):

        '''
        traj[i, 0] : forward direction
        traj[i, 1] : lateral direction
        '''

        r = np.ones(shape=(1, num_grid))
        co = -1 * grid_size * 2
        r_y = co * np.copy(r)
        for i in range(1, num_grid):
            co += grid_size
            r_y = np.concatenate([co*r, r_y], axis=0)
        r_y = np.expand_dims(r_y, axis=2)


        r = np.ones(shape=(num_grid, 1))
        co = grid_size*(num_grid/2)
        r_x = co * np.copy(r)
        for i in range(1, num_grid):
            co -= grid_size
            r_x = np.concatenate([r_x, co*r], axis=1)
        r_x = np.expand_dims(r_x, axis=2)

        return np.concatenate([r_y, r_x], axis=2)

    # update, 220106
    def create_ROI_template_ped(self, grid_size, num_grid):

        '''
        traj[i, 0] : forward direction
        traj[i, 1] : lateral direction

        * r_x : from up (+) to bottom (-)
        [gs*N/2, gs*N/2, gs*N/2, ..., gs*N/2]
        [gs*(N/2-1), ...         ,gs*(N/2-1)]
        .
        .
        [gs*(-N/2), ...           ,gs*(-N/2)]

        * r_y : from left (+) to right (-)
        gs*N/2, gs*(N/2-1), ... , gs*-N/2
        gs*N/2, gs*(N/2-1), ... , gs*-N/2
        .
        .
        gs*N/2, gs*(N/2-1), ... , gs*-N/2
        '''

        r = np.ones(shape=(1, num_grid))
        co = grid_size * (num_grid / 2)

        r_x = co * r
        r_y = co * r.T
        for i in range(1, num_grid):
            co -= grid_size
            r_x = np.concatenate([r_x, co * r], axis=0)
            r_y = np.concatenate([r_y, co * r.T], axis=1)
        r_x = np.expand_dims(r_x, axis=2)
        r_y = np.expand_dims(r_y, axis=2)

        return np.concatenate([r_x, r_y], axis=2)


    def pooling_operation(self, x, feat_map):

        '''
        ROI-pooling feature vectors from feat map

        Inputs
        x : num_agents x 2
        feat_map : 8 x 200 x 200

        Outputs
        pooled_vecs.permute(1, 0) : num_agents x 8
        '''

        ch_dim = feat_map.size(0)

        # from global coordinate system (ego-vehicle coordinate system) to feat_map index
        shift_c = np.trunc(self.y_range_max * self.scale_y)
        shift_r = np.trunc(self.x_range_max * self.scale_x)

        c_pels_f, c_oob = self.clip(-(toNP(x[:, 1]) * self.scale_y) + shift_c)
        r_pels_f, r_oob = self.clip(-(toNP(x[:, 0]) * self.scale_x) + shift_r)
        oob_pels = np.logical_or(c_oob, r_oob)



        # 4 neighboring positions
        '''
        -------|------|
        | lu   | ru   |
        |(cur.)|      |
        |------|------|
        | ld   | rd   |
        |      |      |
        |------|------|
        '''

        c_pels = c_pels_f.astype('int')
        r_pels = r_pels_f.astype('int')

        c_pels_lu = np.copy(c_pels) # j
        r_pels_lu = np.copy(r_pels) # i

        c_pels_ru, _ = self.clip(np.copy(c_pels + 1)) # j + 1
        r_pels_ru, _ = self.clip(np.copy(r_pels)) # i

        c_pels_ld, _ = self.clip(np.copy(c_pels)) # j
        r_pels_ld, _ = self.clip(np.copy(r_pels + 1)) # i + 1

        c_pels_rd, _ = self.clip(np.copy(c_pels + 1)) # j + 1
        r_pels_rd, _ = self.clip(np.copy(r_pels + 1)) # i + 1


        # feats (ch x r x c)
        feat_rd = feat_map[:, r_pels_rd.astype('int'), c_pels_rd.astype('int')]
        feat_lu = feat_map[:, r_pels_lu.astype('int'), c_pels_lu.astype('int')]
        feat_ru = feat_map[:, r_pels_ru.astype('int'), c_pels_ru.astype('int')]
        feat_ld = feat_map[:, r_pels_ld.astype('int'), c_pels_ld.astype('int')]

        # calc weights, debug 210409
        alpha = r_pels_f - r_pels_lu.astype('float')
        beta = c_pels_f - c_pels_lu.astype('float')

        dist_lu = (1 - alpha) * (1 - beta) + 1e-10
        dist_ru = (1 - alpha) * beta
        dist_ld = alpha * (1 - beta)
        dist_rd = alpha * beta

        # weighted sum of features, debug 210409
        w_lu = toTS(dist_lu, dtype=x).view(1, -1).repeat_interleave(ch_dim, dim=0)
        w_ru = toTS(dist_ru, dtype=x).view(1, -1).repeat_interleave(ch_dim, dim=0)
        w_ld = toTS(dist_ld, dtype=x).view(1, -1).repeat_interleave(ch_dim, dim=0)
        w_rd = toTS(dist_rd, dtype=x).view(1, -1).repeat_interleave(ch_dim, dim=0)

        pooled_vecs = (w_lu * feat_lu) + (w_ru * feat_ru) + (w_ld * feat_ld) + (w_rd * feat_rd)
        pooled_vecs[:, oob_pels] = 0

        return pooled_vecs.view(ch_dim, self.roi_num_grid, self.roi_num_grid)

    def forward(self, xo, input_map, R, seq_start_end):

        '''
        xo : observed traj
        input_map : concat of voxels and HDmap or its conv results
        R : rotation matrix
        '''

        # update, 220106
        if (self.model_mode == 'vehicle'):
            roi_template = self.create_ROI_template(self.roi_grid_size, self.roi_num_grid).reshape(self.roi_num_grid ** 2, 2)
        else:
            roi_template = self.create_ROI_template_ped(self.roi_grid_size, self.roi_num_grid).reshape(
                self.roi_num_grid ** 2, 2)

        roi_template = torch.from_numpy(roi_template).to(xo)
        num_scenes = seq_start_end.size(0)

        feature_list = []
        for s in range(num_scenes):
            start = seq_start_end[s, 0].item()
            end = seq_start_end[s, 1].item()
            num_agents = end - start

            cur_traj_scene = xo[:, start:end, :]
            cur_R_scene = R[start:end]
            for a in range(num_agents):

                cur_pos = cur_traj_scene[-1, a, :2].reshape(1, 2)
                cur_R = cur_R_scene[a]
                rroi_index = torch.matmul(cur_R, roi_template.permute(1, 0)).permute(1, 0) + cur_pos

                agent_feature = torch.unsqueeze(self.pooling_operation(rroi_index, input_map[s]), dim=0)
                feature_list.append(agent_feature)

        return torch.cat(feature_list, dim=0) # num_agents x 256 x roi_num_grid x roi_num_grid


class ActorFeatureExtraction(nn.Module):

    def __init__(self, args, onnx=False):
        super(ActorFeatureExtraction, self).__init__()

        self.onnx = onnx
        self.max_num_agents = args.max_num_agents
        self.agent_mapfeat_dim = args.agent_mapfeat_dim
        self.cnn_outch_dim = args.cnn_outch_dim

        layers = []
        layers.append(nn.Conv2d(in_channels=args.cnn_outch_dim, out_channels=args.cnn_outch_dim, kernel_size=3, padding=2, dilation=2, bias=False))
        layers.append(nn.BatchNorm2d(args.cnn_outch_dim))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        layers.append(nn.Conv2d(in_channels=args.cnn_outch_dim, out_channels=args.cnn_outch_dim, kernel_size=3, padding=2, dilation=2, bias=False))
        layers.append(nn.BatchNorm2d(args.cnn_outch_dim))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        layers.append(nn.Conv2d(in_channels=args.cnn_outch_dim, out_channels=args.cnn_outch_dim, kernel_size=3, padding=2, dilation=2, bias=False))
        layers.append(nn.BatchNorm2d(args.cnn_outch_dim))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        layers.append(nn.Conv2d(in_channels=args.cnn_outch_dim, out_channels=self.agent_mapfeat_dim, kernel_size=3, padding=2, dilation=2, bias=False))
        layers.append(nn.BatchNorm2d(self.agent_mapfeat_dim))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        self.merge_layer = nn.Sequential(*layers)

    def forward(self, actor_feat_pooling):

        '''
        actor_feat_pooling : num_agents x 256 x roi_num_grid x roi_num_grid
        '''

        batch = self.max_num_agents if self.onnx else actor_feat_pooling.size(0)

        return self.merge_layer(actor_feat_pooling).view(batch, self.agent_mapfeat_dim)


class Encoder(nn.Module):


    def __init__(self, args):
        super(Encoder, self).__init__()

        self.agent_mapfeat_dim = args.agent_mapfeat_dim # 512
        self.traj_enc_h_dim = args.traj_enc_h_dim # 64
        self.cvae_enc_dim = args.cvae_enc_dim # 64
        self.latent_dim = args.latent_dim

        input_dim = self.agent_mapfeat_dim + 2 * self.traj_enc_h_dim
        self.mean_func = make_mlp([input_dim, self.cvae_enc_dim, self.latent_dim], [True, True], ['relu', 'none'], [0, 0])
        self.logvar_func = make_mlp([input_dim, self.cvae_enc_dim, self.latent_dim], [True, True], ['relu', 'none'], [0, 0])

    def forward(self, context):

        '''
        h0 : agent context, cat(agent_motion_context (past & future), agent_scene_context)
        '''

        mean = self.mean_func(context)
        log_var = self.logvar_func(context)

        return mean, log_var


class Prior(nn.Module):


    def __init__(self, args):
        super(Prior, self).__init__()

        self.agent_mapfeat_dim = args.agent_mapfeat_dim # 512
        self.traj_enc_h_dim = args.traj_enc_h_dim # 64
        self.cvae_enc_dim = args.cvae_enc_dim # 64
        self.latent_dim = args.latent_dim

        input_dim = self.agent_mapfeat_dim + self.traj_enc_h_dim
        self.mean_func = make_mlp([input_dim, self.cvae_enc_dim, self.latent_dim], [True, True], ['relu', 'none'], [0, 0])
        self.logvar_func = make_mlp([input_dim, self.cvae_enc_dim, self.latent_dim], [True, True], ['relu', 'none'], [0, 0])

    def forward(self, context):

        '''
        h0 : agent context, cat(agent_motion_context (past & future), agent_scene_context)
        '''

        mean = self.mean_func(context)
        log_var = self.logvar_func(context)

        return mean, log_var


class TrajDecoder(nn.Module):


    def __init__(self, args, onnx=False):
        super(TrajDecoder, self).__init__()

        self.onnx = onnx
        self.max_num_agents = args.max_num_agents
        self.best_k = args.best_k

        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.traj_dec_h_dim = args.traj_dec_h_dim
        self.traj_enc_h_dim = args.traj_enc_h_dim
        self.latent_dim = args.latent_dim
        self.agent_mapfeat_dim = args.agent_mapfeat_dim # 512
        self.dec_drop_prob = args.dec_drop_prob

        self.merge_layer0 = make_mlp([self.agent_mapfeat_dim, 128, 64], [True, True], ['relu', 'None'], [0, 0])
        self.z_decoder = make_mlp([self.latent_dim+self.traj_enc_h_dim, 64], [True], ['None'], [0])
        self.traj_decoder = make_mlp([64, 64, self.pred_len*2], [True, True], ['relu', 'None'], [args.dec_drop_prob, 0])

    def forward(self, agent_past_motion_context, agent_scene_context, Z):

        num_total_agents = self.max_num_agents * self.best_k if self.onnx else agent_past_motion_context.size(0)

        merge0 = self.merge_layer0(agent_scene_context)
        scale = F.softmax(self.z_decoder(torch.cat((agent_past_motion_context, Z), dim=1)), dim=1)
        pred_traj_ego = self.traj_decoder(scale * merge0)
        return pred_traj_ego.view(num_total_agents, self.pred_len, 2)


class Scratch(nn.Module):


    def __init__(self, args, onnx=False, best_k=10):
        super(Scratch, self).__init__()

        self.onnx = onnx
        self.max_num_agents = args.max_num_agents
        self.best_k = best_k

        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.latent_dim = args.latent_dim

        # define past traj encoder, update, 220511
        if (args.traj_enc_type == 0):
            self.past_traj_enc = TrajEncoder(args=args, input_dim=2, onnx=onnx)
            self.future_traj_enc = TrajEncoder(args=args, input_dim=2, onnx=onnx, is_obs=False)
        else:
            self.past_traj_enc = TrajResEncoder(args=args, input_dim=2, onnx=onnx)
            self.future_traj_enc = TrajResEncoder(args=args, input_dim=2, onnx=onnx, is_obs=False)


        # spatial context extractor
        self.AFE = ActorFeatureExtraction(args=args, onnx=onnx)

        # cvae encoder/prior
        self.Encoder = Encoder(args=args)
        self.Prior = Prior(args=args)

        # trajectory decoder
        self.Traj_Dec = TrajDecoder(args=args, onnx=onnx)

        print(">> Model is loaded from {%s} " % os.path.basename(__file__))

    def reparameterize(self, mu, logvar, eps):
        std = torch.exp(0.5 * logvar)
        return eps.mul(std).add_(mu)

    def forward(self, obs_traj, obs_traj_a, agent_feat_pool, Rm, Z):

        '''
        obs_traj : obs_len x num_total_agents x 2 (global coordinate system = ego-vehicle coordinate system)
        agent_feat_pool : num_scenes x ch
        R : num_total_agents x 2 x 2 (Rotation matrix for traj)
        Z : (num_max_agents x best_k) x lantent_dim
        '''

        best_k = self.best_k

        # encode trajectories
        agent_past_motion_context = self.past_traj_enc(obs_traj_a)[0]

        # extract actor context
        agent_scene_context = self.AFE(agent_feat_pool)

        # CVAE_prior
        cat_enc = torch.cat((agent_past_motion_context.repeat(best_k, 1), agent_scene_context.repeat(best_k, 1)), dim=1)
        mean, log_var = self.Prior(cat_enc)
        Z_reparam = self.reparameterize(mean, log_var, Z)


        # SIM decoder
        # cat_dec = torch.cat((agent_past_motion_context.repeat(best_k, 1),
        #                      agent_scene_context.repeat(best_k, 1),
        #                      Z_reparam), dim=1)


        pred_trajs_a = self.Traj_Dec(agent_past_motion_context.repeat(best_k, 1),
                                     agent_scene_context.repeat(best_k, 1),
                                     Z_reparam) # (batch x best_k) x pred_len x 2

        return pred_trajs_a


    def inference(self, obs_traj, obs_traj_a, agent_feat_pool, Rm, best_k_in):

        '''
        obs_traj : obs_len x num_total_agents x 2 (global coordinate system = ego-vehicle coordinate system)
        agent_feat_pool : num_scenes x ch
        Rt : num_total_agents x 2 x 2 (Rotation matrix for traj)
        best_k_in : (1)
        '''

        if (self.onnx == False):
            batch = obs_traj_a.size(1)
            best_k = best_k_in
        else:
            batch = self.max_num_agents
            best_k = self.best_k

        # encode trajectories
        agent_past_motion_context = self.past_traj_enc(obs_traj_a)[0]

        # extract actor context
        agent_scene_context = self.AFE(agent_feat_pool)


        Zs = torch.randn(size=(best_k*batch, self.latent_dim)).to(obs_traj)
        # Zs = torch.ones(size=(best_k * batch, self.latent_dim)).to(obs_traj)

        # CVAE_prior
        cat_enc = torch.cat((agent_past_motion_context.repeat(best_k, 1), agent_scene_context.repeat(best_k, 1)), dim=1)
        mean, log_var = self.Prior(cat_enc)
        Z_reparam = self.reparameterize(mean, log_var, Zs)

        # cat_dec = torch.cat((agent_past_motion_context.repeat(best_k, 1),
        #                      agent_scene_context.repeat(best_k, 1),
        #                      Z_reparam), dim=1)
        pred_trajs_a = self.Traj_Dec(agent_past_motion_context.repeat(best_k, 1),
                                     agent_scene_context.repeat(best_k, 1),
                                     Z_reparam) # (batch x best_k) x pred_len x 2

        # transform to global coordinate system
        center_pos = obs_traj[self.obs_len-1].repeat(best_k, 1)
        pred_trajs_e = torch.bmm(Rm.repeat(best_k, 1, 1), pred_trajs_a.permute(0, 2, 1)).permute(2, 0, 1) + center_pos

        return pred_trajs_e, pred_trajs_a, Zs

