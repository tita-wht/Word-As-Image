import collections.abc
import os
import os.path as osp
from torch import nn
import kornia.augmentation as K
import pydiffvg
import save_svg
import cv2
from ttf import font_string_to_svgs, normalize_letter_size
import torch
import numpy as np
from scipy.integrate import quad

def edict_2_dict(x):
    if isinstance(x, dict):
        xnew = {}
        for k in x:
            xnew[k] = edict_2_dict(x[k])
        return xnew
    elif isinstance(x, list):
        xnew = []
        for i in range(len(x)):
            xnew.append( edict_2_dict(x[i]))
        return xnew
    else:
        return x


def check_and_create_dir(path):
    pathdir = osp.split(path)[0]
    if osp.isdir(pathdir):
        pass
    else:
        os.makedirs(pathdir)


def update(d, u):
    """https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def preprocess(font, word, letter, level_of_cc=1):

    if level_of_cc == 0:
        target_cp = None
    else:
        target_cp = {"A": 120, "B": 120, "C": 100, "D": 100,
                     "E": 120, "F": 120, "G": 120, "H": 120,
                     "I": 35, "J": 80, "K": 100, "L": 80,
                     "M": 100, "N": 100, "O": 100, "P": 120,
                     "Q": 120, "R": 130, "S": 110, "T": 90,
                     "U": 100, "V": 100, "W": 100, "X": 130,
                     "Y": 120, "Z": 120,
                     "a": 120, "b": 120, "c": 100, "d": 100,
                     "e": 120, "f": 120, "g": 120, "h": 120,
                     "i": 35, "j": 80, "k": 100, "l": 80,
                     "m": 100, "n": 100, "o": 100, "p": 120,
                     "q": 120, "r": 130, "s": 110, "t": 90,
                     "u": 100, "v": 100, "w": 100, "x": 130,
                     "y": 120, "z": 120
                     }
        target_cp = {k: v * level_of_cc for k, v in target_cp.items()}

    print(f"======= {font} =======")
    font_path = f"code/data/fonts/{font}.ttf"
    init_path = f"code/data/init"
    subdivision_thresh = None
    font_string_to_svgs(init_path, font_path, word, target_control=target_cp,
                        subdivision_thresh=subdivision_thresh)
    normalize_letter_size(init_path, font_path, word)

    # optimaize two adjacent letters
    if len(letter) > 1:
        subdivision_thresh = None
        font_string_to_svgs(init_path, font_path, letter, target_control=target_cp,
                            subdivision_thresh=subdivision_thresh)
        normalize_letter_size(init_path, font_path, letter)

    print("Done preprocess")


def get_data_augs(cut_size):
    augmentations = []
    augmentations.append(K.RandomPerspective(distortion_scale=0.5, p=0.7))
    augmentations.append(K.RandomCrop(size=(cut_size, cut_size), pad_if_needed=True, padding_mode='reflect', p=1.0))
    return nn.Sequential(*augmentations)


'''pytorch adaptation of https://github.com/google/mipnerf'''
def learning_rate_decay(step,
                        lr_init,
                        lr_final,
                        max_steps,
                        lr_delay_steps=0,
                        lr_delay_mult=1):
  """Continuous learning rate decay function.
  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
  is log-linearly interpolated elsewhere (equivalent to exponential decay).
  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
  function of lr_delay_mult, such that the initial learning rate is
  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
  to the normal learning rate when steps>lr_delay_steps.
  Args:
    step: int, the current optimization step.
    lr_init: float, the initial learning rate.
    lr_final: float, the final learning rate.
    max_steps: int, the number of steps during optimization.
    lr_delay_steps: int, the number of steps to delay the full learning rate.
    lr_delay_mult: float, the multiplier on the rate when delaying it.
  Returns:
    lr: the learning for current step 'step'.
  """
  if lr_delay_steps > 0:
    # A kind of reverse cosine decay.
    delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
        0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
  else:
    delay_rate = 1.
  t = np.clip(step / max_steps, 0, 1)
  log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
  return delay_rate * log_lerp



def save_image(img, filename, gamma=1):
    check_and_create_dir(filename)
    imshow = img.detach().cpu()
    pydiffvg.imwrite(imshow, filename, gamma=gamma)


def get_letter_ids(letter, word, shape_groups):
    for group, l in zip(shape_groups, word):
        if l == letter:
            return group.shape_ids


def combine_word(word, letter, font, experiment_dir):
    word_svg_scaled = f"./code/data/init/{font}_{word}_scaled.svg"
    canvas_width_word, canvas_height_word, shapes_word, shape_groups_word = pydiffvg.svg_to_scene(word_svg_scaled)
    letter_ids = []
    for l in letter:
        letter_ids += get_letter_ids(l, word, shape_groups_word)

    w_min, w_max = min([torch.min(shapes_word[ids].points[:, 0]) for ids in letter_ids]), max(
        [torch.max(shapes_word[ids].points[:, 0]) for ids in letter_ids])
    h_min, h_max = min([torch.min(shapes_word[ids].points[:, 1]) for ids in letter_ids]), max(
        [torch.max(shapes_word[ids].points[:, 1]) for ids in letter_ids])

    c_w = (-w_min + w_max) / 2
    c_h = (-h_min + h_max) / 2

    svg_result = os.path.join(experiment_dir, "output-svg", "output.svg")
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(svg_result)

    out_w_min, out_w_max = min([torch.min(p.points[:, 0]) for p in shapes]), max(
        [torch.max(p.points[:, 0]) for p in shapes])
    out_h_min, out_h_max = min([torch.min(p.points[:, 1]) for p in shapes]), max(
        [torch.max(p.points[:, 1]) for p in shapes])

    out_c_w = (-out_w_min + out_w_max) / 2
    out_c_h = (-out_h_min + out_h_max) / 2

    scale_canvas_w = (w_max - w_min) / (out_w_max - out_w_min)
    scale_canvas_h = (h_max - h_min) / (out_h_max - out_h_min)

    if scale_canvas_h > scale_canvas_w:
        wsize = int((out_w_max - out_w_min) * scale_canvas_h)
        scale_canvas_w = wsize / (out_w_max - out_w_min)
        shift_w = -out_c_w * scale_canvas_w + c_w
    else:
        hsize = int((out_h_max - out_h_min) * scale_canvas_w)
        scale_canvas_h = hsize / (out_h_max - out_h_min)
        shift_h = -out_c_h * scale_canvas_h + c_h

    for num, p in enumerate(shapes):
        p.points[:, 0] = p.points[:, 0] * scale_canvas_w
        p.points[:, 1] = p.points[:, 1] * scale_canvas_h
        if scale_canvas_h > scale_canvas_w:
            p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min + shift_w
            p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min
        else:
            p.points[:, 0] = p.points[:, 0] - out_w_min * scale_canvas_w + w_min
            p.points[:, 1] = p.points[:, 1] - out_h_min * scale_canvas_h + h_min + shift_h

    for j, s in enumerate(letter_ids):
        shapes_word[s] = shapes[j]

    save_svg.save_svg(
        f"{experiment_dir}/{font}_{word}_{letter}.svg", canvas_width, canvas_height, shapes_word,
        shape_groups_word)

    render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes_word, shape_groups_word)
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + \
               torch.ones(img.shape[0], img.shape[1], 3, device="cuda:0") * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    save_image(img, f"{experiment_dir}/{font}_{word}_{letter}.png")


def create_video(num_iter, experiment_dir, video_frame_freq):
    img_array = []
    for ii in range(0, num_iter):
        if ii % video_frame_freq == 0 or ii == num_iter - 1:
            filename = os.path.join(
                experiment_dir, "video-png", f"iter{ii:04d}.png")
            img = cv2.imread(filename)
            img_array.append(img)

    video_name = os.path.join(
        experiment_dir, "video.mp4")
    check_and_create_dir(video_name)
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (600, 600))
    for iii in range(len(img_array)):
        out.write(img_array[iii])
    out.release()


def clean_paths(shapes, shape_groups, threshold=0.3):
    print("clean paths: ", len(shapes))
    for i, shape in enumerate(shapes):
        if shape_groups[i].fill_color[3] < threshold:
            del shapes[i]
            j = i
            while j < len(shape_groups): 
                shape_group = shape_groups[j]
                new_ids = [x-1 for x in shape_group.shape_ids]
                new_ids = torch.tensor(new_ids)
                shape_group.shape_ids = new_ids
                j += 1
            del shape_groups[i]
            i -= 1
    print("num_paths: ", len(shapes))
    return shapes, shape_groups

def clean_paths_by_shape(shapes, shape_groups, threshold=0.3):
    print("clean paths: ", len(shapes))
    del_ids = []
    f = 0
    for shape in shapes:
        if f==0:
            print(shape.num_control_points)
            print(shape.points)
            f = 1
    print("-> num_paths: ", len(shapes))


def clean_paths_by_raster(w,h, shapes, shape_groups, threshold=0.003, b=0.2):
    print("clean paths by raster: ", len(shapes), end = " ")
    render = pydiffvg.RenderFunction.apply
    device = pydiffvg.get_device()
    bg = torch.zeros(w,h,4, device=device)
    scene_args = pydiffvg.RenderFunction.serialize_scene(w,h,shapes,shape_groups)
    img = render(w,h,2,2,0,None,*scene_args)

    # path_area_masks = torch.empty((0,w,h,4),device=device)
    dists = [] # path一つの画面支配率
    path_imgs = []
    for shape_group in shape_groups:
        sgs = [shape_group]
        scene_args = pydiffvg.RenderFunction.serialize_scene(w,h,shapes,sgs)
        single_path_img = render(w,h,2,2,0,bg,*scene_args)
        path_imgs.append(single_path_img)

        path_area_mask = torch.where(single_path_img!=bg, 1, 0) # （背景色と比較して）path領域内を1,領域外を0のマスクを生成
        path_area_mask = torch.sum(path_area_mask, dim=2)
        path_area_mask = torch.where(path_area_mask>=1.0, 1, 0)

        # path_area_masks = torch.cat((path_area_masks, path_area_mask.unsqueeze(0)),dim=0)

        # 面積が小さく、不透明度が低いものを削除したい。
        dist = torch.sum(path_area_mask)/(w*h) # 面積
        # dist = dist * shape_group.fill_color[3]
        gain = 20 # 大きいほど閾値以下の不透明度領域をカット。
        bias = b # 不透明度の閾値
        dist = dist * torch.sigmoid((shape_group.fill_color[3]-bias)*gain) # sigmoidによって不透明度を0.3周りで活性化
        dists.append(dist.item())
    # single_path_img = single_path_img.detach().cpu()
    # pydiffvg.imwrite(single_path_img, "tmp.png", gamma=1.0)
    dists = torch.tensor(dists)

    # ソート情報が必要なとき
    # dists = torch.tensor(dists)
    # sorted_indices = torch.argsort(dists)
    # sorted_values = dists[sorted_indices]

    # 面積+色ベース(dist)によって閾値以下のパスを削除する。
    i = 0
    while i<len(shapes):
        if dists[i]<threshold:
            del shapes[i]
            del path_imgs[i]
            j = i
            while j<len(shape_groups):
                shape_group = shape_groups[j]
                new_ids = [x-1 for x in shape_group.shape_ids]
                new_ids = torch.tensor(new_ids)
                shape_group.shape_ids = new_ids
                j += 1
            del shape_groups[i]
            dists = torch.cat((dists[:i], dists[i+1:]),dim=0)
            i -= 1 # 消した次の要素を飛ばすのを予防
        i += 1
    print("-> ", len(shapes))
    # shapes, shape_groups = clean_behind_paths(shapes, shape_groups, path_imgs)
    return shapes, shape_groups

def clean_behind_paths(w,h,shapes, shape_groups, threshold=5e-4):
    # パスとその前景の不透明度より、パスがどの程度投下されているのかを用いてパスを削除
    # 1e-3,5e-3,1e-4,5e-4,5e-5
    print("clean_befind_paths:", len(shapes), end=" ")
    render = pydiffvg.RenderFunction.apply
    device = pydiffvg.get_device()
    bg = torch.zeros(w,h,4, device=device)
    scene_args = pydiffvg.RenderFunction.serialize_scene(w,h,shapes,shape_groups)

    # path_area_masks = torch.empty((0,w,h,4),device=device)
    path_imgs = []
    nnnn = len(shapes)
    for shape_group in shape_groups:
        sgs = [shape_group]
        scene_args = pydiffvg.RenderFunction.serialize_scene(w,h,shapes,sgs)
        single_path_img = render(w,h,2,2,0,bg,*scene_args)
        path_imgs.append(single_path_img)

    i = 0 
    leaks = []
    while i < len(shapes)-1:
        cur = _mix_paths((path_imgs[i:]))
        foreground = _mix_paths(path_imgs[i+1:])
        leak = cur - foreground # path[i]が見えている部分(不透明度)
        leak = torch.sum(leak**2)/(w*h)
        # print(leak.item())
        leaks.append(leak.item())
        # print(leak.item())
        if leak < threshold:
            del shapes[i]
            del path_imgs[i]
            j = i
            while j<len(shape_groups):
                shape_group = shape_groups[j]
                new_ids = [x-1 for x in shape_group.shape_ids]
                new_ids = torch.tensor(new_ids)
                shape_group.shape_ids = new_ids
                j += 1
            del shape_groups[i]
            i -= 1 # 消した次の要素を飛ばすのを予防
        i += 1
    # with open(f'output_{nnnn}.txt', 'w') as file:
    #     for l in leaks:
    #         file.write(str(l) + '\n')
    print("->", len(shapes))
    return shapes, shape_groups
    
def _mix_paths(path_imgs):
    # 最終不透明度を計算混色する（レンダラの機能分解)
    w,h,_ = path_imgs[0].size()
    pre_a = path_imgs[0][:,:,3:4]
    if len(path_imgs)==1: 
        return pre_a

    for i in range(1, len(path_imgs)):
        cur_a = path_imgs[i][:,:,3:4] + pre_a*(1-path_imgs[i][:,:,3:4])
        pre_a = cur_a
    return cur_a



def get_shape_mask(w, h, shapes, shape_groups, masking_value=1e-9):
    # おそらく負の方向に値が飛んで行っていると思われる
    render = pydiffvg.RenderFunction.apply
    device = pydiffvg.get_device()
    bg = torch.zeros(w,h,4, device=device)
    scene_args = pydiffvg.RenderFunction.serialize_scene(w,h,shapes,shape_groups)
    img = render(w,h,2,2,0,None,*scene_args)

    scene_args = pydiffvg.RenderFunction.serialize_scene(w,h,shapes,shape_groups)
    single_path_img = render(w,h,2,2,0,bg,*scene_args)

    path_area_mask = torch.where(single_path_img!=bg, 1, 0) # （背景色と比較して）path領域内を1,領域外を0のマスクを生成
    path_area_mask = torch.sum(path_area_mask, dim=2)
    path_area_mask = torch.where(path_area_mask>=1.0, 1, 0)
    
    # path_mask = torch.sum(path_area_mask, dim=0)
    # path_mask = torch.where(path_mask>=1.0, 0, 1) # 背景部分をマスク1とする。
    path_area_mask = path_area_mask * masking_value

    return path_area_mask

def inclease_segments(shapes, magnification=2):
    # デ・カステロのアルゴリズムを用いてパスを分割(セグメントを増加)
    print("inclease segments:", shapes[0].points.size()[0], end=" ")
    new_shapes = []
    # pathは閉じているものと仮定
    for s, shape in enumerate(shapes):
        ncp = shape.num_control_points
        points = shape.points
        new_ncp = torch.zeros(len(ncp)*magnification, dtype=torch.int32) + 2
        new_points = points[0].detach().clone().unsqueeze(0)
        with torch.no_grad():
            for i in range(len(ncp)):
                ctrls = points[i*3: i*3+3]
                if i != len(ncp)-1:
                    ctrls = torch.cat((ctrls, points[i*3+3].unsqueeze(0)),dim=0)
                else:
                    ctrls = torch.cat((ctrls, points[0].unsqueeze(0)),dim=0)
                for j in range(magnification-1):
                    ctrls0, ctrls1 = _deCasteljau(ctrls, 1.0*(j+1)/magnification)
                    new_points = torch.cat((new_points, ctrls0[1:]),dim=0)
                    new_points = torch.cat((new_points, ctrls1[1:]),dim=0)
            new_points = new_points[:-1] # 最後の要素=最初の要素
            new_points.requires_grad = True
            # print(points)
            # print(new_points)
            """ 分割後のアンカー2つの距離を測り距離が一定以上なら分割する """

        new_shape = pydiffvg.Path(
            num_control_points = new_ncp,
            points = new_points,
            stroke_width = shape.stroke_width,
            is_closed = shape.is_closed,
        )
        new_shapes.append(new_shape)
    # print(new_ncp.size())
    # print("new_poitns size: ", new_points.size())
    # print("dasa", new_points.requires_grad, new_points.is_leaf)
    print("-> ", shapes[0].points.size()[0])
    return new_shapes

def _derivative(t, P):
    dx_dt = -3*(1-t)**2 * P[0] + 3*(1-t)**2 * P[1] - 6*(1-t)*t * P[1] - 3*t**2 * P[2] + 6*(1-t)*t * P[2] + 3*t**2 * P[3]
    dy_dt = -3*(1-t)**2 * P[0] + 3*(1-t) * t**2 * P[0] - 6*(1-t)*t * P[1] + 6*(1-t)*t * P[2] - 3*t**2 * P[2] + 3*t**2 * P[3]
    return np.sqrt(dx_dt**2 + dy_dt**2)
def bezier_curve_length(P):
    # ベジェ曲線の曲線長(近似)
    length, _ = quad(lambda t: _derivative(t, P), 0, 1)
    return length

def _deCasteljau(points, t):
    # t: div rate
    ctrls0 = [points[0]]
    ctrls1 = [points[-1]]
    num_p = points.size()[0]
    while num_p > 1:
        points = _get_dividing_points(points,t)
        ctrls0.append(points[0])
        ctrls1.append(points[-1])
        num_p = num_p - 1
    ctrls0 = torch.stack(ctrls0)
    ctrls1 = torch.stack(ctrls1[::-1]) # reverse
    return ctrls0, ctrls1
def _get_dividing_points(points, t):
    # print(points)
    n = points.size()[0]
    if n == 1: 
        return points
    new_points = []
    for i in range(n-1):
        new_points.append((points[i+1]-points[i])*t+points[i])
    new_points = torch.stack(new_points)
    return new_points
