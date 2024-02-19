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
from losses import SDSLoss, ToneLoss, ConformalLoss, DistanceTransformLoss
from config import set_config
from utils import *
from xing_loss import xing_loss
import wandb
import warnings

def init_shapes(svg_path, trainable: Mapping[str, bool]): # 追加

    svg = f'{svg_path}.svg'
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg)

    parameters = edict()

    # path points
    if trainable.point:
        parameters.point = []
        parameters.color = []
        for path in shapes_init:
            path.points.requires_grad = True
            parameters.point.append(path.points)
        for shape_group in shape_groups_init: # 追加
            shape_group.fill_color.requires_grad = True
            parameters.color.append(shape_group.fill_color)

    return shapes_init, shape_groups_init, parameters

def init_point(img, num_stroke): # 追加
    """Return init points

    Args:
        img (PIL.Image): Input Image
        num_stroke (np.array): Init points

    Returns:

    """
    img = np.array(img)
    points = np.where(img == 0.0)
    index = np.random.choice(list(range(len(points[0]))), num_stroke, replace=True)
    # plt.scatter(600-points[1][index],600-points[0][index])
    # plt.axis("off")
    # plt.savefig("init_points.png")
    return points, index

def init_curves(num_paths, use_closed, init_points, index, canvas_width, canvas_height, placement="circle"):
    """_summary_

    Args:
        opt (_type_): _description_
        init_points (_type_): _description_
        index (_type_): _description_
        canvas_width (_type_): _description_
        canvas_height (_type_): _description_

    Returns:
        _type_: _description_
    """
    shapes = []
    shape_groups = []

    for i in range(num_paths):
        # num_segments = random.randint(3, 5) if use_closed else random.randint(1, 3)
        num_segments = 6 # 4
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        radius = 0.2 #if placement=="random" else random.uniform(0.01,0.1)
        # 下位のパスが隠れないように大→小になるように初期化(或いは小さい領域のパスを最適化の途中で追加)

        points = []
        p0 = (
            float(init_points[1][index[i]] / canvas_width),
            float(init_points[0][index[i]] / canvas_height),
        )
        if placement == "circle":
            p00 = p0
            # radius = random.uniform(0.2,1.0) * radius
            radius = 0.05
            p0 = _tmp_rotate2(p00, 0, radius)
        points.append(p0)
        for j in range(num_segments):
            if placement == "random":
                p1 = (
                    p0[0] + radius * (random.random() - 0.5),
                    p0[1] + radius * (random.random() - 0.5),
                )
                p2 = (
                    p1[0] + radius * (random.random() - 0.5),
                    p1[1] + radius * (random.random() - 0.5),
                )
                p3 = (
                    p2[0] + radius * (random.random() - 0.5),
                    p2[1] + radius * (random.random() - 0.5),
                )
            elif placement == "circle":
                p1 = _tmp_rotate2(p00, 360 * (j*3+1)/(num_segments*3), radius) #random.uniform(0.01,1.0) * radius)
                p2 = _tmp_rotate2(p00, 360 * (j*3+2)/(num_segments*3), radius) #random.uniform(0.01,1.0) * radius)
                p3 = _tmp_rotate2(p00, 360 * (j*3+3)/(num_segments*3), radius) #random.uniform(0.01,1.0) * radius)
                

            points.append(p1)
            points.append(p2)
            if j < num_segments - 1:
                points.append(p3)
                p0 = p3
        # print(points)
        if placement=="random":
            points = torch.tensor(points)
        elif placement=="circle":
            points = torch.stack(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(
            num_control_points=num_control_points,
            points=points,
            stroke_width=torch.tensor(1.0),
            is_closed=use_closed,
        )
        shapes.append(path)

        if use_closed:
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=torch.tensor([random.random(), random.random(), random.random(), 1.0]) #random.uniform(0.7,1.0)]),# random.random()]),
            )
        else:
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=torch.tensor([random.random(), random.random(), random.random(), random.random()]),
            )
        shape_groups.append(path_group)

    return shapes, shape_groups

def init_p0(num_strokes, canvas_width, canvas_height, init_shape:torch.tensor=None):
    """ 
    num_strokes個のベジェ曲線の始点位置を初期化する。
    init_shapeがある場合、その黒色領域内に始点位置を限定する。
    """
    if init_shape is not None:
        init_shape = init_shape.cpu().numpy()
        init_points = np.where(init_shape[0] == 0.0)
    else:
        init_points = (
            np.random.randint(0, canvas_height, num_strokes),
            np.random.randint(0, canvas_width, num_strokes)
        )
    index = np.random.choice(list(range(len(init_points[0]))), num_strokes, replace=True)
    p0 = torch.empty(0,2)
    for i in range(num_strokes):
        xy = torch.tensor([
            float(init_points[1][index[i]] / canvas_width),
            float(init_points[0][index[i]] / canvas_height), ])
        xy = xy.unsqueeze(0)
        p0 = torch.cat((p0, xy))
    return p0 # [num_strokes, 2:[p0x,p0y]]
def _tmp_rotate(points, theta, radius):
    # pointsに対して、points_x-radiusを中心としてtheta回転する
    # points: [b, strokes, (x,y)] 
    # radius:[b,strokes,1(radius)]の場合
    rad = math.radians(theta)
    # p = torch.tensor([radius, 0.0])
    p = torch.cat((radius, torch.zeros_like(radius)),dim=2)
    p_new = torch.empty_like(p)
    p_new[...,0] = p[...,0]*math.cos(rad) - p[...,1]*math.sin(rad)
    p_new[...,1] = p[...,0]*math.sin(rad) + p[...,1]*math.cos(rad) 
    points_new = points.clone().detach()
    points_new[:,:,0] = points_new[:,:,0] + p_new[:,:,0] - radius
    points_new[:,:,1] = points_new[:,:,1] + p_new[:,:,1]
    return points_new

def _tmp_rotate2(points, theta, radius):
    # pointsに対して、points_x-radiusを中心としてtheta回転する
    # points: [b, strokes, (x,y)] 
    # radius:[b,strokes,1(radius)]の場合
    # radius = radius * random.uniform(0.01,1.0)
    radius = 0.03
    points = torch.tensor(points)
    rad = math.radians(theta)
    p = torch.tensor([radius, radius])
    p_new = torch.empty_like(p)
    p_new[0] = p[0]*math.cos(rad) - p[1]*math.sin(rad)
    p_new[1] = p[0]*math.sin(rad) + p[1]*math.cos(rad) 
    points_new = points.clone().detach()
    points_new[0] = points_new[0] + p_new[0]
    points_new[1] = points_new[1] + p_new[1]
    return points_new

def init_paths(batch_size, num_strokes, num_segments, canvas_width, canvas_height, init_shape=None, use_closed=True):
    # 円版
    # init points for bezier path
    p0 = torch.empty(0,num_strokes,2)
    for b in range(batch_size):
        p = init_p0(num_strokes,canvas_width,canvas_height,init_shape=init_shape)
        p0 = torch.cat((p0,p.unsqueeze(0))) # → [batch, num_strokes, 2]
    points = p0.unsqueeze(2) # [batch, num_strokes, 1, 2]
    
    # radius = 0.1
    radius = torch.rand((batch_size, num_strokes, 1))*0.1
    for i in range(num_segments*3-1):
        p = _tmp_rotate(p0, 360 * (i+1)/(num_segments*3), radius)
        points = torch.cat((points, p.unsqueeze(2)), dim=2) # → [b, strokes, num_segments, 2]
        points[:,:,i,0] = points[:,:,i,0] + radius[:,:,0]
    points[:,:,num_segments*3-1,0] = points[:,:,num_segments*3-1,0] + radius[:,:,0]
    points[:,:,:,0] = points[:,:,:,0] * canvas_width
    points[:,:,:,1] = points[:,:,:,1] * canvas_height
        
    # init line-width
    if use_closed == True:
        stroke_widths = torch.ones(batch_size,num_strokes)
    else:
        stroke_widths = torch.rand(batch_size, num_strokes) * 10
    
    colors = torch.rand(batch_size, num_strokes, 4)
  
    # definition of paths / shape_group
    shapes = []
    shape_groups = []
    num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
    
    for b in range(batch_size):
        path = [pydiffvg.Path(
            num_control_points = num_control_points,
            points = points[b,i],
            stroke_width = stroke_widths[b,i],
            is_closed = use_closed) for i in range(num_strokes)]
        shapes.append(path) # → list[[paths_list(0)],...,[paths_list(batchs-1)]]
        
        if use_closed:
            shape_group = [pydiffvg.ShapeGroup(
                shape_ids = torch.tensor([i]),
                fill_color = colors[b,i]) for i in range(num_strokes)]
        else:
            shape_group = [pydiffvg.ShapeGroup(                
                shape_ids = torch.tensor([i]),
                fill_color = None,
                stroke_color = colors[b,i]) for i in range(num_strokes)]
        shape_groups.append(shape_group) # → like shapes
        
    return shapes, shape_groups

