"""
python code/main.py  --semantic_concept "LEAVES" --word "NATURE" --optimized_letter "T" --font "HobeauxRococeaux-Sherman" 
--seed 0

python code/main.py  --semantic_concept "bunny" --target_file "star" --device "cuda:1"

python code/main.py  --semantic_concept "bunny" --target_file "star"  --use_wandb 1 --wandb_user y_ryuta1301
"""

from typing import Mapping
import os
import numpy as np
import random
from tqdm import tqdm
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg
import save_svg
import math
from losses import SDSLoss, ToneLoss, ConformalLoss, DistanceTransformLoss, CircumferenceLoss
from config import set_config
from utils import *
from init_func import *
from xing_loss import xing_loss
import wandb
import warnings
warnings.filterwarnings("ignore")

pydiffvg.set_print_timing(False)
gamma = 1.0


if __name__ == "__main__":
    
    cfg = set_config()
    num_paths = cfg.num_paths

    # use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    pydiffvg.set_device(cfg.device)
    device = pydiffvg.get_device()
    print("device: ", device)

    print("preprocessing")
    svg = pydiffvg.svg_to_scene(cfg.target+".svg") # (svg_w,svg_h,[Shapes],[ShapeGroups])
    if cfg.font != "none":
        preprocess(cfg.font, cfg.word, cfg.optimized_letter, cfg.level_of_cc) # ここ

    h, w = cfg.render_size, cfg.render_size 

    data_augs = get_data_augs(cfg.cut_size)

    render = pydiffvg.RenderFunction.apply

    """"""
    # 利用するsvg画像に対して白黒のマスキングpngを作成,保存する
    sargs = pydiffvg.RenderFunction.serialize_scene(*svg)
    init_raster = render(h, w, 2, 2, 0, None, *sargs)
    init_raster = init_raster[:, :, 3:4] * init_raster[:, :, :3] + \
               torch.ones(init_raster.shape[0], init_raster.shape[1], 3, device=device) * (1 - init_raster[:, :, 3:4])
    init_raster = init_raster[:, :, :3]
    init_raster_mask = torch.ones_like(init_raster) != init_raster
    init_raster[init_raster_mask] = 0
    save_image(init_raster, cfg.target+".png", gamma) 
    
    init_mask = transforms.ToPILImage()(init_raster.permute((2,0,1))) # PIL画像化
    print("init_mask size: ", init_mask.size)

    """"""
    ### 一部だけを切り抜き最適化 ###
    path_dist_rate = 0.2 if cfg.crop_part else 0.0
    print("path_dist_rate: ", path_dist_rate)
    if cfg.crop_part:
        crop_area = (w/4,h/4,w*3/4,h*3/4) # 右下で仮置き
        cropped_mask = init_mask.crop(crop_area)
        new_mask = Image.new("RGB", init_mask.size, "white")
        new_mask.paste(cropped_mask,(int(crop_area[0]),int(crop_area[1])))
        print("cropped_mask size: ", cropped_mask.size)
        print("new_mask size: ", new_mask.size)
        # new_mask.save("./tmp.jpg") # 最適化範囲を確認したい場合
        # init_mask = new_mask
        c_points, c_index = init_point(new_mask, int(num_paths*path_dist_rate))
        c_shapes, c_shape_groups = init_curves(int(num_paths*path_dist_rate), True, c_points, c_index, w, h)


    ### initialize shape ###
    print('initializing shape')
    # shapes, shape_groups, parameters = init_shapes(svg_path=cfg.target, trainable=cfg.trainable)
    i_points, i_index = init_point(init_mask, int(num_paths*(1-path_dist_rate)))
    shapes, shape_groups = init_curves(int(num_paths*(1-path_dist_rate)), True, i_points, i_index, w, h) # ここ
    
    """ crop有りの場合 """
    # shapes = shapes+c_shapes
    # shape_groups = shape_groups+c_shape_groups
    # shapes = c_shapes
    # shape_groups = c_shape_groups

    """"""
    parameters = edict() # parametersを別建て
    # path points　requires_grad=True
    if True:
        parameters.point = []
        parameters.color = []
        for path in shapes:
            path.points.requires_grad = True
            parameters.point.append(path.points)
        for shape_group in shape_groups:
            shape_group.fill_color.requires_grad = True
            parameters.color.append(shape_group.fill_color)
        if cfg.crop_part:
            for path in c_shapes:
                path.points.requires_grad = True
                parameters.point.append(path.points)
            for shape_group in c_shape_groups:
                shape_group.fill_color.requires_grad = True
                parameters.color.append(shape_group.fill_color)

    nnn = 20
    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups[:nnn])

    """"""
    ### 背景画像読み込み ###
    bg_img = transforms.ToTensor()(Image.open(cfg.target+".png")).permute(1,2,0)
    bg_img = bg_img.to(device)
    print("bg: ",cfg.bg)
    print("bg size: ", bg_img.size()) # [600,600,3]

    
    """"""
    ### 初期画像を設定 ###
    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + \
               torch.ones(img_init.shape[0], img_init.shape[1], 3, device=device) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]

    if cfg.use_wandb: # 初期画像をwandbに追加
        plt.imshow(img_init.detach().cpu())
        wandb.log({"init": wandb.Image(plt)}, step=0)
        plt.close()

    """"""
    ### SDS loss ###
    if cfg.loss.use_sds_loss:
        sds_loss = SDSLoss(cfg, device)
    ### tone loss ###
    if cfg.loss.tone.use_tone_loss:
        tone_loss = ToneLoss(cfg)
        tone_loss.set_image_init(bg_img)
    ### conformal loss ###
    if cfg.loss.conformal.use_conformal_loss:
        conformal_loss = ConformalLoss(parameters, device, cfg.optimized_letter, shape_groups)
    ### distance transform loss ###
    if cfg.loss.dt.use_dt_loss:
        dt_loss = DistanceTransformLoss(init_mask)

    
    """"""
    if cfg.save.init:
        print('saving init')
        filename = os.path.join(
            cfg.experiment_dir, "svg-init", "init.svg")
        check_and_create_dir(filename)
        save_svg.save_svg(filename, w, h, shapes, shape_groups)

    """"""
    ### loop setting ###    
    num_iter = cfg.num_iter
    pg = [{'params': parameters["point"], 'lr': cfg.lr_base["point"]}]
    pg_color = [{'params': parameters["color"], 'lr': 1e-3}]
    optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)
    optim_color = torch.optim.Adam(pg_color, betas=(0.9,0.9), eps=1e-6)

    """"""
    ### 学習率スケジューラの設定 ###
    lr_lambda = lambda step: learning_rate_decay(
        step, 
        cfg.lr.lr_init, 
        cfg.lr.lr_final, 
        num_iter,     
        lr_delay_steps=cfg.lr.lr_delay_steps,
        lr_delay_mult=cfg.lr.lr_delay_mult) / cfg.lr.lr_init
    lr_lambda_color = lambda step: learning_rate_decay(
        step, 
        cfg.lr_color.lr_init, 
        cfg.lr_color.lr_final, 
        num_iter,     
        lr_delay_steps=cfg.lr_color.lr_delay_steps,
        lr_delay_mult=cfg.lr_color.lr_delay_mult) / cfg.lr_color.lr_init
    scheduler = LambdaLR(optim, lr_lambda=lr_lambda, last_epoch=-1)  # lr.base * lrlambda_f
    scheduler_color = LambdaLR(optim_color, lr_lambda=lr_lambda_color, last_epoch=-1)

    total_segments = len(shapes)*shapes[0].num_control_points.size()[0]
    """"""
    ### training loop ###
    print("start training")
    t_range = tqdm(range(num_iter))
    for step in t_range:
        # print(shapes[0].points.size())
        if cfg.use_wandb:
            wandb.log({"learning_rate": optim.param_groups[0]['lr']}, step=step)
        optim.zero_grad()
        optim_color.zero_grad()

        # render image
        # for i in range(len(shapes)):
            # print(torch.find(shapes[i].points==NaN,))
        bg_noise = torch.rand((w,h,4), device=device) # 白色部分を背景でごまかすの防止用
        if nnn<len(shape_groups): nnn = nnn + len(shape_groups)*2/num_iter
        # print(nnn)
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups[:int(nnn)])
        img = render(w, h, 2, 2, step, bg_noise, *scene_args)

        # compose image with white background
        if cfg.loss.tone.use_tone_loss:
            # x_gray = (img == bg_noise).float()
            # x_gray = torch.where(x_gray!=bg_noise, 0.0, 1.0)
            # x_gray.requires_grad = True
            x_gray = (img - bg_noise)**2
            x_gray = torch.sigmoid(x_gray*1e6) # 0以外を1.0
            x_gray = 1.0 - x_gray

            x_gray = x_gray.permute(2,0,1).unsqueeze(0)
            x_gray = x_gray[:,:3,:,:]
            # x_gray = torch.sum(x_gray,dim=1,keepdim=True)/4

        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]

        if cfg.save.video and (step % cfg.save.video_frame_freq == 0 or step == num_iter - 1):
            save_image(img, os.path.join(cfg.experiment_dir, "video-png", f"iter{step:04d}.png"), gamma)
            filename = os.path.join(
                cfg.experiment_dir, "video-svg", f"iter{step:04d}.svg")
            check_and_create_dir(filename)
            save_svg.save_svg(
                filename, w, h, shapes, shape_groups)
            if cfg.use_wandb:
                plt.imshow(img.detach().cpu())
                wandb.log({"img": wandb.Image(plt)}, step=step)
                plt.close()

        # shape_mask = get_shape_mask(w,h,shapes,shape_groups, masking_value=1) 
        # img = img * shape_mask.unsqueeze(2)
        x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        x = x.repeat(cfg.batch_size, 1, 1, 1)
        x_aug = data_augs.forward(x)


        # compute diffusion loss per pixel
        loss = sds_loss(x_aug)
        loss = loss + CircumferenceLoss.calc_loss(shapes) * cfg.loss.xing.xing_w
        # loss = tone_loss(x_gray)
        if cfg.use_wandb:
            wandb.log({"sds_loss": loss.item()}, step=step)
        if cfg.loss.tone.use_tone_loss:
            tone_loss_res = tone_loss(x_gray)#, step)
            if cfg.use_wandb:
                wandb.log({"dist_loss": tone_loss_res}, step=step)
            loss = loss + tone_loss_res
        # if cfg.loss.conformal.use_conformal_loss:
        #     loss_angles = conformal_loss()
        #     loss_angles = cfg.loss.conformal.angeles_w * loss_angles
        #     if cfg.use_wandb:
        #         wandb.log({"loss_angles": loss_angles}, step=step)
        #     loss = loss + loss_angles
        if cfg.loss.dt.use_dt_loss:
            dt_l = dt_loss(img)
            loss = loss + dt_l * cfg.loss.dt.dt_w
            if cfg.use_wandb:
                wandb.log({"dt_loss": dt_l}, step=step)
        if cfg.loss.xing.use_xing_loss:
            xing_l = xing_loss(parameters["point"])
            loss = loss + xing_l * cfg.loss.xing.xing_w
            if cfg.use_wandb:
                wandb.log({"xing_loss": xing_l}, step=step)

        if cfg.use_wandb:
            wandb.log({"total loss": loss}, step=step)

        t_range.set_postfix({'loss': loss.item()})
        loss.backward()
        optim.step()
        optim_color.step()
        scheduler.step()
        scheduler_color.step()
        if step > num_iter/2 and step%(num_iter/5) == 0: 
            # 要らないパスを減らす
            # 必要なパスのセグメントを増やす
            shapes, shape_groups = clean_paths_by_raster(w,h,shapes, shape_groups, threshold=0.003, b=0.3)
            # if total_segments/len(shapes)*shapes[0].num_control_points.size()[0]>2.0:
            #     shapes = inclease_segments(shapes)
            #     total_segments = len(shapes)*shapes[0].num_control_points.size()[0]
            para_pt = []
            for path in shapes:
                path.points.requres_grad = True
                para_pt.append(path.points)
            pg = [{'params': para_pt, 'lr': cfg.lr_base["point"]}]
            optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)
            scheduler = LambdaLR(optim, lr_lambda=lr_lambda, last_epoch=-1)  # lr.base * lrlambda_f
            

    """"""
    ### 最終画像を保存 ###
    filename = os.path.join(
        cfg.experiment_dir, "output-svg", "output.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(
        filename, w, h, shapes, shape_groups)

    if cfg.optimized_letter is not None:
        combine_word(cfg.word, cfg.optimized_letter, cfg.font, cfg.experiment_dir)

    if cfg.save.image:
        filename = os.path.join(
            cfg.experiment_dir, "output-png", "output.png")
        check_and_create_dir(filename)
        # shapes, shape_groups = clean_paths(shapes,shape_groups)
        # clean_paths_by_shape(shapes,shape_groups) ## ここ
        # shapes, shape_groups = clean_paths_by_raster(w,h,shapes,shape_groups)
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        if cfg.bg: # bg_imgにカラーを付加
            # bg_img = -(1.0 - bg_img) # reverse 
            bg_img = torch.cat((bg_img,torch.ones_like((bg_img[:,:,0]), device=device).unsqueeze(2)),dim=2)
            bg_color = torch.tensor((random.random(),random.random(),random.random(),1.0), device=device)
            print("bg_color: ", bg_color)
            bg_img[bg_img[:,:,0]==0.0] = bg_color 
            # bg_img = bg_img.to(device)
            img = render(w, h, 2, 2, step, bg_img, *scene_args)
        else:
            img = render(w, h, 2,2, step, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (1 - img[:, :, 3:4])
        imshow = img[:, :, :3].detach().cpu()
        pydiffvg.imwrite(imshow, filename, gamma=gamma)
        if cfg.use_wandb:
            plt.imshow(img.detach().cpu())
            wandb.log({"img": wandb.Image(plt)}, step=step)
            plt.close()

    if cfg.save.video:
        print("saving video")
        create_video(cfg.num_iter, cfg.experiment_dir, cfg.save.video_frame_freq)

    if cfg.use_wandb:
        wandb.finish()
