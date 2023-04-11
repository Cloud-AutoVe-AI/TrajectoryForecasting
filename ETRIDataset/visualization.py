from utils.functions import *
from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box

class Visualizer:

    def __init__(self, args, map, x_range, y_range, z_range, map_size, obs_len, pred_len):

        self.args = args
        self.map = map

        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.map_size = map_size

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        self.scale_y = float(map_size - 1) / axis_range_y
        self.scale_x = float(map_size - 1) / axis_range_x

        self.obs_len = int(args.past_horizon_seconds * args.target_sample_period)
        self.pred_len = int(args.future_horizon_seconds * args.target_sample_period)

        self.dpi = 80
        self.color_centerline = (0, 0, 0)
        self.color_roadmark = 'yellowgreen'
        self.color_laneside = (0.5, 0.5, 0.5)


        self.palette = make_palette(pred_len)



    # -------------------------------
    # TOPVIEW HDMAP
    # -------------------------------
    def draw_centerlines(self, ax, pose, x_range, y_range, map_size):

        def draw_lines_on_topview(ax, line, x_range, y_range, map_size, color, ID=None):

            axis_range_y = y_range[1] - y_range[0]
            axis_range_x = x_range[1] - x_range[0]
            scale_y = float(map_size - 1) / axis_range_y
            scale_x = float(map_size - 1) / axis_range_x

            col_pels = -(line[:, 1] * scale_y).astype(np.int32)
            row_pels = -(line[:, 0] * scale_x).astype(np.int32)

            col_pels += int(np.trunc(y_range[1] * scale_y))
            row_pels += int(np.trunc(x_range[1] * scale_x))

            ax.plot(col_pels, self.map_size - row_pels, '-', linewidth=1.0, color=color, alpha=1)
            # if (ID is not None):
            #     lane_len = int(line.shape[0] / 2)
            #     ax.text(self.map_size - row_pels[lane_len], col_pels[lane_len], "ID{}".format(cur_ID), fontsize=10)
            return ax

        lane_segs, IDs = self.map.return_centerlines(pose, x_range, y_range)
        for l in range(len(lane_segs)):

            cur_lane = lane_segs[l]
            # cur_ID = IDs[l]
            ax = draw_lines_on_topview(ax, cur_lane[:, :2], x_range, y_range, map_size=map_size, color=self.color_centerline)


        return ax

    def draw_laneside(self, ax, pose, x_range, y_range, map_size):

        def draw_lines_on_topview(ax, line, x_range, y_range, map_size, color):

            axis_range_y = y_range[1] - y_range[0]
            axis_range_x = x_range[1] - x_range[0]
            scale_y = float(map_size - 1) / axis_range_y
            scale_x = float(map_size - 1) / axis_range_x

            col_pels = -(line[:, 1] * scale_y).astype(np.int32)
            row_pels = -(line[:, 0] * scale_x).astype(np.int32)

            col_pels += int(np.trunc(y_range[1] * scale_y))
            row_pels += int(np.trunc(x_range[1] * scale_x))

            ax.plot(col_pels, self.map_size - row_pels, '-', linewidth=1.0, color=color, alpha=1)

            return ax

        lane_segs = self.map.return_laneside(pose, x_range, y_range)
        for l in range(len(lane_segs)):

            cur_lane = lane_segs[l]
            ax = draw_lines_on_topview(ax, cur_lane[:, :2], x_range, y_range, map_size=map_size, color=self.color_laneside)

        return ax

    def draw_polygons(self, ax, pose, x_range, y_range, map_size, facecolor, alpha):

        axis_range_y = y_range[1] - y_range[0]
        axis_range_x = x_range[1] - x_range[0]
        scale_y = float(map_size - 1) / axis_range_y
        scale_x = float(map_size - 1) / axis_range_x

        poly_segs = self.map.return_roadmark(pose, x_range, y_range)

        for i in range(len(poly_segs)):
            cur_polygons = poly_segs[i]

            col_pels = -(cur_polygons[:, 1] * scale_y).astype(np.int32)
            row_pels = -(cur_polygons[:, 0] * scale_x).astype(np.int32)

            col_pels += int(np.trunc(self.y_range[1] * scale_y))
            row_pels += int(np.trunc(self.x_range[1] * scale_x))

            col_pels[col_pels < 0] = 0
            col_pels[col_pels > self.map_size - 1] = self.map_size - 1

            row_pels[row_pels < 0] = 0
            row_pels[row_pels > self.map_size - 1] = self.map_size - 1

            contours = np.concatenate(
                [col_pels.reshape(cur_polygons.shape[0], 1), self.map_size - row_pels.reshape(cur_polygons.shape[0], 1)], axis=1)

            ax.add_patch(
                patches.Polygon(
                    contours,
                    closed=True,
                    facecolor=facecolor,
                    alpha=alpha
                ))

        #lightgray
        return ax

    def topview_hdmap(self, ax, pose, x_range, y_range, map_size):

        ax = self.draw_centerlines(ax, pose, x_range, y_range, map_size)

        ax = self.draw_laneside(ax, pose, x_range, y_range, map_size)

        ax = self.draw_polygons(ax, pose, x_range, y_range, map_size, self.color_roadmark, alpha=0.5)

        return ax


    # -------------------------------
    # BBOX and TRAJECTORY
    # -------------------------------

    def topview_bbox(self, ax, agent, color):


        bbox = agent.bbox_e

        # to topview image domain
        col_pels = -(bbox[:, 1] * self.scale_y).astype(np.int32)
        row_pels = -(bbox[:, 0] * self.scale_x).astype(np.int32)

        col_pels += int(np.trunc(self.y_range[1] * self.scale_y))
        row_pels += int(np.trunc(self.x_range[1] * self.scale_x))

        row_pels = self.map_size - row_pels

        line_col = [col_pels[0], col_pels[1]]
        line_row = [row_pels[0], row_pels[1]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[1], col_pels[2]]
        line_row = [row_pels[1], row_pels[2]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[2], col_pels[3]]
        line_row = [row_pels[2], row_pels[3]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[3], col_pels[0]]
        line_row = [row_pels[3], row_pels[0]]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        col_pel = int((col_pels[0] + col_pels[3]) / 2)
        row_pel = int((row_pels[0] + row_pels[3]) / 2)

        line_col = [col_pels[1], col_pel]
        line_row = [row_pels[1], row_pel]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        line_col = [col_pels[2], col_pel]
        line_row = [row_pels[2], row_pel]
        ax.plot(line_col, line_row, '-', linewidth=1.0, color=color, alpha=1)

        text = str(int(agent.track_id))
        ax.text(col_pel, row_pel, text)

        return ax

    def topview_trajectory(self, ax, gt, pred):
        '''
        gt : seq_len x 2
        '''

        if (len(pred) > 0):
            # Predicted trajs ----------------------------------
            col_pels = -(pred[:, 1] * self.scale_y).astype(np.int32)
            row_pels = -(pred[:, 0] * self.scale_x).astype(np.int32)

            col_pels += int(np.trunc(self.y_range[1] * self.scale_y))
            row_pels += int(np.trunc(self.x_range[1] * self.scale_x))
            row_pels = self.map_size - row_pels

            for t in range(self.pred_len):
                r, g, b = self.palette[t]
                ax.plot(col_pels[t], row_pels[t], 's', linewidth=1.0, color=(r, g, b), alpha=1.0)

        # GT trajs --------------------------------------
        col_pels = -(gt[:, 1] * self.scale_y).astype(np.int32)
        row_pels = -(gt[:, 0] * self.scale_x).astype(np.int32)

        col_pels += int(np.trunc(self.y_range[1] * self.scale_y))
        row_pels += int(np.trunc(self.x_range[1] * self.scale_x))
        row_pels = self.map_size - row_pels

        # ax.plot(col_pels[self.obs_len - 1:], row_pels[self.obs_len - 1:], 'o-', linewidth=1.0, color=(1, 1, 1), alpha=1)
        ax.plot(col_pels[self.obs_len - 1:], row_pels[self.obs_len - 1:], 'o-', linewidth=1.0, color=(0, 0, 0), alpha=0.5)
        ax.plot(col_pels[:self.obs_len], row_pels[:self.obs_len], 'o', linewidth=1.0, color=(0.5, 0.5, 0.5), alpha=0.5)

        return ax

    def topview_traj_distribution(self, ax, pred):

        best_k, pred_len, _ = pred.shape

        # for displaying images
        axis_range_y = self.y_range[1] - self.y_range[0]
        axis_range_x = self.x_range[1] - self.x_range[0]
        scale_y = float(self.map_size - 1) / axis_range_y
        scale_x = float(self.map_size - 1) / axis_range_x

        # for s in range(pred_len):
        #     cur_t = pred[:, s, :]
        #
        #     col_pels = -(cur_t[:, 1] * scale_y)
        #     row_pels = -(cur_t[:, 0] * scale_x)
        #
        #     col_pels += y_range[1] * scale_y
        #     row_pels += x_range[1] * scale_x
        #     row_pels = map_size - row_pels
        #
        #     sns.kdeplot(x=col_pels, y=row_pels, cmap="Reds", shade=True, bw_adjust=.5, ax=ax)

        pred_ = pred.reshape(best_k * pred_len, 2)
        col_pels = -(pred_[:, 1] * scale_y)
        row_pels = -(pred_[:, 0] * scale_x)

        col_pels += self.y_range[1] * scale_y
        row_pels += self.x_range[1] * scale_x
        row_pels = self.map_size - row_pels
        sns.kdeplot(x=col_pels, y=row_pels, cmap="Reds", shade=True, bw_adjust=.5, ax=ax)

        return ax

    def fig_to_nparray(self, fig, ax):

        fig.set_size_inches(self.map_size / self.dpi, self.map_size / self.dpi)
        ax.set_axis_off()

        fig.canvas.draw()
        render_fig = np.array(fig.canvas.renderer._renderer)

        final_img = np.zeros_like(render_fig[:, :, :3]) # 450, 1600
        final_img[:, :, 2] = render_fig[:, :, 0]
        final_img[:, :, 1] = render_fig[:, :, 1]
        final_img[:, :, 0] = render_fig[:, :, 2]

        plt.close()

        return final_img




def read_and_resize_img(path, img_size):
    img = cv2.imread(path)
    return cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)

def in_range_points(points, x, y, z, x_range, y_range, z_range):

    points_select = points[np.logical_and.reduce((x > x_range[0], x < x_range[1], y > y_range[0], y < y_range[1], z > z_range[0], z < z_range[1]))]
    return np.around(points_select, decimals=2)

def make_palette(pred_len):
    red = np.array([1, 0, 0])
    orange = np.array([1, 0.5, 0])
    yellow = np.array([1, 1.0, 0])
    green = np.array([0, 1.0, 0])
    blue = np.array([0, 0, 1])
    colors = [red, orange, yellow, green, blue]

    palette = []
    for t in range(pred_len):

        cur_pos = 4.0 * float(t) / float(pred_len - 1)  # pred_len -> 0 ~ 4
        prev_pos = int(cur_pos)
        next_pos = int(cur_pos) + 1

        if (next_pos > 4):
            next_pos = 4

        prev_color = colors[prev_pos]
        next_color = colors[next_pos]

        prev_w = float(next_pos) - cur_pos
        next_w = 1 - prev_w

        cur_color = prev_w * prev_color + next_w * next_color
        palette.append(cur_color)

    return palette

def correspondance_check(win_min_max, lane_min_max):

    # four points for window and lane box
    w_x_min, w_y_min, w_x_max, w_y_max = win_min_max
    l_x_min, l_y_min, l_x_max, l_y_max = lane_min_max

    w_TL = (w_x_min, w_y_max)  # l1
    w_BR = (w_x_max, w_y_min)  # r1

    l_TL = (l_x_min, l_y_max)  # l2
    l_BR = (l_x_max, l_y_min)  # r2

    # If one rectangle is on left side of other
    # if (l1.x > r2.x | | l2.x > r1.x)
    if (w_TL[0] > l_BR[0] or l_TL[0] > w_BR[0]):
        return False

    # If one rectangle is above other
    # if (l1.y < r2.y || l2.y < r1.y)
    if (w_TL[1] < l_BR[1] or l_TL[1] < w_BR[1]):
        return False

    return True

def make_rot_matrix_from_yaw(yaw):

    '''
    yaw in radian
    '''

    m_cos = np.cos(yaw)
    m_sin = np.sin(yaw)
    m_R = [[m_cos, -1 * m_sin], [m_sin, m_cos]]

    return np.array(m_R)

# ------------------------------------
# For debugging
# ------------------------------------

def whl_to_bbox_topview(wlh):

    w, l, h = wlh
    x, y = 0, 0

    x_front = x + (l / 2)
    x_back = x - (l / 2)
    y_left = y + (w / 2)
    y_right = y - (w / 2)

    bbox = [[x_front, y_left], [x_back, y_left], [x_back, y_right], [x_front, y_right], [x_front, y]]

    return np.array(bbox)

def draw_bbox_on_topview(img, bbox, x_range, y_range, map_size, color):

    '''

    bbox (4 x 2)

       (bottom)          (up)

        front              front
    b0 -------- b3    b4 -------- b7
       |      |          |      |
       |      |          |      |
       |      |          |      |
       |      |          |      |
    b1 -------- b2    b5 -------- b6
         rear              rear
    '''


    # for displaying images
    axis_range_y = y_range[1] - y_range[0]
    axis_range_x = x_range[1] - x_range[0]
    scale_y = float(map_size - 1) / axis_range_y
    scale_x = float(map_size - 1) / axis_range_x

    # to topview image domain
    col_pels = -(bbox[:, 1] * scale_y).astype(np.int32)
    row_pels = -(bbox[:, 0] * scale_x).astype(np.int32)

    col_pels += int(np.trunc(y_range[1] * scale_y))
    row_pels += int(np.trunc(x_range[1] * scale_x))

    cv2.line(img, (col_pels[0], row_pels[0]), (col_pels[1], row_pels[1]), color, 1)
    cv2.line(img, (col_pels[1], row_pels[1]), (col_pels[2], row_pels[2]), color, 1)
    cv2.line(img, (col_pels[2], row_pels[2]), (col_pels[3], row_pels[3]), color, 1)
    cv2.line(img, (col_pels[3], row_pels[3]), (col_pels[0], row_pels[0]), color, 1)

    col_pel = int((col_pels[0] + col_pels[3]) / 2)
    row_pel = int((row_pels[0] + row_pels[3]) / 2)
    cv2.line(img, (col_pels[1], row_pels[1]), (col_pel, row_pel), color, 1)
    cv2.line(img, (col_pels[2], row_pels[2]), (col_pel, row_pel), color, 1)

    return img

def draw_traj_on_topview(img, traj, obs_len, x_range, y_range, map_size, in_color):
    '''
    traj : seq_len x 2
    '''

    # for displaying images
    axis_range_y = y_range[1] - y_range[0]
    axis_range_x = x_range[1] - x_range[0]
    scale_y = float(map_size - 1) / axis_range_y
    scale_x = float(map_size - 1) / axis_range_x

    # GT trajs --------------------------------------
    col_pels = -(traj[:, 1] * scale_y).astype(np.int32)
    row_pels = -(traj[:, 0] * scale_x).astype(np.int32)

    col_pels += int(np.trunc(y_range[1] * scale_y))
    row_pels += int(np.trunc(x_range[1] * scale_x))

    for j in range(0, traj.shape[0]):
        if (traj[j, 0] != -1000):
            color = in_color
            if (j < obs_len):
                color = (64, 64, 64)
            cv2.circle(img, (col_pels[j], row_pels[j]), 2, color, -1)

    return img

def clip(array, map_size):
    OOB = np.logical_or((array < 0), (array > map_size - 1))

    array[array < 0] = 0
    array[array > map_size - 1] = map_size - 1

    return array, OOB

def create_ROI(grid_size, num_grid):
    '''
    traj[i, 0] : y-axis (forward direction)
    traj[i, 1] : x-axis (lateral direction)
    '''
    # grid_size = 0.2
    # num_grid = 100

    r = np.ones(shape=(1, num_grid))
    co = -4
    r_y = co * np.copy(r)
    for i in range(1, num_grid):
        co += grid_size
        r_y = np.concatenate([co * r, r_y], axis=0)
    r_y = np.expand_dims(r_y, axis=2)

    r = np.ones(shape=(num_grid, 1))
    co = grid_size * (num_grid / 2)
    r_x = co * np.copy(r)
    for i in range(1, num_grid):
        co -= grid_size
        r_x = np.concatenate([r_x, co * r], axis=1)
    r_x = np.expand_dims(r_x, axis=2)

    return np.concatenate([r_y, r_x], axis=2)

def create_ROI_ped(grid_size, num_grid):
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

def pooling_operation(img, x, x_range, y_range, map_size):
    '''
    ROI-pooling feature vectors from feat map

    Inputs
    x : num_agents x 2
    feat_map : 8 x 200 x 200

    Outputs
    pooled_vecs.permute(1, 0) : num_agents x 8
    '''

    x_range_max = x_range[1]
    y_range_max = y_range[1]

    axis_range_y = y_range[1] - y_range[0]
    axis_range_x = x_range[1] - x_range[0]
    scale_y = float(map_size - 1) / axis_range_y
    scale_x = float(map_size - 1) / axis_range_x

    # from global coordinate system (ego-vehicle coordinate system) to feat_map index
    shift_c = np.trunc(y_range_max * scale_y)
    shift_r = np.trunc(x_range_max * scale_x)

    c_pels_f, c_oob = clip(-(x[:, 1] * scale_y) + shift_c, map_size)
    r_pels_f, r_oob = clip(-(x[:, 0] * scale_x) + shift_r, map_size)
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

    c_pels_lu = np.copy(c_pels)
    r_pels_lu = np.copy(r_pels)

    c_pels_ru, _ = clip(np.copy(c_pels + 1), map_size)
    r_pels_ru, _ = clip(np.copy(r_pels), map_size)

    c_pels_ld, _ = clip(np.copy(c_pels), map_size)
    r_pels_ld, _ = clip(np.copy(r_pels + 1), map_size)

    c_pels_rd, _ = clip(np.copy(c_pels + 1), map_size)
    r_pels_rd, _ = clip(np.copy(r_pels + 1), map_size)

    # feats (ch x r x c)
    feat_rd = img[r_pels_rd.astype('int'), c_pels_rd.astype('int'), :]
    feat_lu = img[r_pels_lu.astype('int'), c_pels_lu.astype('int'), :]
    feat_ru = img[r_pels_ru.astype('int'), c_pels_ru.astype('int'), :]
    feat_ld = img[r_pels_ld.astype('int'), c_pels_ld.astype('int'), :]

    # calc weights, debug 210409
    alpha = r_pels_f - r_pels_lu.astype('float')
    beta = c_pels_f - c_pels_lu.astype('float')

    dist_lu = (1 - alpha) * (1 - beta) + 1e-10
    dist_ru = (1 - alpha) * beta + 1e-10
    dist_ld = alpha * (1 - beta) + 1e-10
    dist_rd = alpha * beta

    # weighted sum of features, debug 210409
    ch_dim = 3
    w_lu = toTS(dist_lu, dtype=torch.float).view(1, -1).repeat_interleave(ch_dim, dim=0)
    w_ru = toTS(dist_ru, dtype=torch.float).view(1, -1).repeat_interleave(ch_dim, dim=0)
    w_ld = toTS(dist_ld, dtype=torch.float).view(1, -1).repeat_interleave(ch_dim, dim=0)
    w_rd = toTS(dist_rd, dtype=torch.float).view(1, -1).repeat_interleave(ch_dim, dim=0)

    w_lu = toNP(w_lu).T
    w_ru = toNP(w_ru).T
    w_ld = toNP(w_ld).T
    w_rd = toNP(w_rd).T

    pooled_vecs = (w_lu * feat_lu) + (w_ru * feat_ru) + (w_ld * feat_ld) + (w_rd * feat_rd)
    pooled_vecs[oob_pels] = 0

    return pooled_vecs
