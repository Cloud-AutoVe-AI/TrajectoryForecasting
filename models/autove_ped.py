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
        self.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # update, 220106
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


class ActorFeatureExtraction(nn.Module):

    def __init__(self, args):
        super(ActorFeatureExtraction, self).__init__()

        # agent's related params
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
        self.cnn_outch_dim = args.cnn_outch_dim

        # aggregation cnn, update according to paper, 210323
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

    def clip(self, array):

        OOB = np.logical_or((array < 0), (array > self.map_size - 1))

        array[array < 0] = 0
        array[array > self.map_size - 1] = self.map_size - 1

        return array, OOB

    def create_ROI_template(self, grid_size, num_grid):

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

        roi_template = self.create_ROI_template(self.roi_grid_size, self.roi_num_grid).reshape(self.roi_num_grid ** 2, 2)
        roi_template = torch.from_numpy(roi_template).to(xo)
        num_scenes = seq_start_end.size(0)
        num_total_agents = xo.size(1)

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

        agent_features_cat = torch.cat(feature_list, dim=0) # num_agents x 256 x roi_num_grid x roi_num_grid
        agent_context_vectors = self.merge_layer(agent_features_cat).view(num_total_agents, self.agent_mapfeat_dim)

        return agent_context_vectors


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


    def __init__(self, args):
        super(TrajDecoder, self).__init__()

        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.traj_dec_h_dim = args.traj_dec_h_dim
        self.traj_enc_h_dim = args.traj_enc_h_dim
        self.latent_dim = args.latent_dim
        self.agent_mapfeat_dim = args.agent_mapfeat_dim # 512
        self.dec_drop_prob = args.dec_drop_prob

        # traj decoder
        input_dim = self.agent_mapfeat_dim+self.traj_enc_h_dim+self.latent_dim
        self.decoder_mlp = make_mlp([input_dim, 128, self.pred_len*2], [True, True], ['relu', 'None'], [args.dec_drop_prob, 0])

    def forward(self, context):

        '''
        h0 : agent context, cat(agent_motion_context, agent_scene_context)
        '''
        num_total_agents = context.size(0)
        pred_traj_ego = self.decoder_mlp(context).view(num_total_agents, self.pred_len, 2)

        return pred_traj_ego


class Scratch(nn.Module):


    def __init__(self, args):
        super(Scratch, self).__init__()

        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)
        self.latent_dim = args.latent_dim

        # define past traj encoder
        self.past_traj_enc = TrajEncoder(args=args, input_dim=2)
        self.future_traj_enc = TrajEncoder(args=args, input_dim=2, is_obs=False)

        # spatial context extractor
        self.AFE = ActorFeatureExtraction(args=args)

        # cvae encoder & prior
        self.Encoder = Encoder(args=args)
        self.Prior = Prior(args=args)

        # trajectory decoder
        self.Traj_Dec = TrajDecoder(args=args)

        print(">> Model is loaded from {%s} " % os.path.basename(__file__))


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, obs_traj, future_traj, obs_traj_a, future_traj_a, seq_start_end, feat_map, Rm, Rt, best_k):

        '''
        obs_traj : obs_len x num_total_agents x 2 (global coordinate system = ego-vehicle coordinate system)
        future_traj : pred_len x num_total_agents x 2
        seq_start_end : num_scenes x 2
        feam_map : num_scenes x ch x h x w
        Rm = num_total_agents x 2 x 2 (Rotation matrix for featmap)
        Rt = num_total_agents x 2 x 2 (Rotation matrix for traj)
        '''

        # encode trajectories
        agent_past_motion_context = self.past_traj_enc(obs_traj_a)[0]
        agent_future_motion_context = self.future_traj_enc(future_traj_a)[0]

        # extract actor context
        agent_scene_context = self.AFE(obs_traj, feat_map, Rm, seq_start_end)

        # CVAE_encoder
        cat_enc = torch.cat((agent_past_motion_context, agent_scene_context, agent_future_motion_context), dim=1)
        mean1, log_var1 = self.Encoder(cat_enc)

        # CVAE_prior
        cat_prior = torch.cat((agent_past_motion_context, agent_scene_context), dim=1)
        mean2, log_var2 = self.Prior(cat_prior)

        # decoder
        pred_trajs_ego = []
        for k in range(best_k):
            Z = self.reparameterize(mean1, log_var1)
            cat_dec = torch.cat((agent_past_motion_context, agent_scene_context, Z), dim=1)
            pred_trajs_ego.append(self.Traj_Dec(cat_dec))

        return pred_trajs_ego, mean1, log_var1, mean2, log_var2

    def inference(self, obs_traj, obs_traj_a, seq_start_end, feat_map, Rm, Rt, best_k):

        '''
        obs_traj : obs_len x num_total_agents x 2 (global coordinate system = ego-vehicle coordinate system)
        future_traj : pred_len x num_total_agents x 2
        seq_start_end : num_scenes x 2
        feam_map : num_scenes x ch x h x w
        Rm = num_total_agents x 2 x 2 (Rotation matrix for featmap)
        Rt = num_total_agents x 2 x 2 (Rotation matrix for traj)
        '''

        batch = obs_traj_a.size(1)

        # encode trajectories
        agent_past_motion_context = self.past_traj_enc(obs_traj_a)[0]

        # extract actor context
        agent_scene_context = self.AFE(obs_traj, feat_map, Rm, seq_start_end)

        # CVAE_prior
        cat_prior = torch.cat((agent_past_motion_context, agent_scene_context), dim=1)
        mean2, log_var2 = self.Prior(cat_prior)

        # SIM decoder
        pred_trajs = []
        for k in range(best_k):
            Z = self.reparameterize(mean2, log_var2)
            cat_dec = torch.cat((agent_past_motion_context, agent_scene_context, Z), dim=1)

            # transform to global coordinate system
            center_pos = obs_traj[-1, :, :]
            pred_traj_ego = self.Traj_Dec(cat_dec)
            pred_traj = torch.bmm(torch.inverse(Rt), pred_traj_ego.permute(0, 2, 1)).permute(2, 0, 1) + center_pos
            pred_trajs.append(pred_traj)

        return pred_trajs

