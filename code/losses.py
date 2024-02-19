import torch.nn as nn
import torchvision
from torchvision import transforms
from scipy.spatial import Delaunay
import torch
import numpy as np
from torch.nn import functional as nnf
from easydict import EasyDict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2

from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
import pydiffvg
import clip

# from utils import save_image
from PIL import Image

class SDSLoss(nn.Module):
    def __init__(self, cfg, device):
        super(SDSLoss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(cfg.diffusion.model,
                                                       torch_dtype=torch.float16, use_auth_token=cfg.token)
        self.pipe = self.pipe.to(self.device)
        # print(help(self.pipe))
        # default scheduler: PNDMScheduler(beta_start=0.00085, beta_end=0.012,
        # beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        self.text_embeddings = None
        self.embed_text()

        # self.tar_img = None
        # self.img_embedding = None
        # self.embed_image()

    def embed_text(self):
        # tokenizer and embed text
        text_input = self.pipe.tokenizer(self.cfg.caption, padding="max_length",
                                         max_length=self.pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
        uncond_input = self.pipe.tokenizer([""], padding="max_length",
                                         max_length=text_input.input_ids.shape[-1],
                                         return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
            # img_embedding = self.pipe.encoder
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        self.text_embeddings = self.text_embeddings.repeat_interleave(self.cfg.batch_size, 0)
        # print(text_embeddings.size())
        del self.pipe.tokenizer
        del self.pipe.text_encoder

    def forward(self, x_aug):
        sds_loss = 0
        # encode rendered image
        x = x_aug * 2. - 1.
        with torch.cuda.amp.autocast():
            init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample())
        latent_z = 0.18215 * init_latent_z  # scaling_factor * init_latents

        with torch.inference_mode():
            # sample timesteps
            timestep = torch.randint(
                low=50,
                high=min(950, self.cfg.diffusion.timesteps) - 1,  # avoid highest timestep | diffusion.timesteps=1000
                size=(latent_z.shape[0],),
                device=self.device, dtype=torch.long)

            # add noise
            eps = torch.randn_like(latent_z)
            # zt = alpha_t * latent_z + sigma_t * eps
            noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

            # denoise
            z_in = torch.cat([noised_latent_zt] * 2)  # expand latents for classifier free guidance
            timestep_in = torch.cat([timestep] * 2)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps_t_uncond, eps_t = self.pipe.unet(z_in, timestep_in, encoder_hidden_states=self.text_embeddings).sample.float().chunk(2)

            # print(eps_t_uncond.size(), self.cfg.diffusion.guidance_scale, eps_t.size(), eps_t_uncond.size())
            eps_t = eps_t_uncond + self.cfg.diffusion.guidance_scale * (eps_t - eps_t_uncond)

            # w = alphas[timestep]^0.5 * (1 - alphas[timestep]) = alphas[timestep]^0.5 * sigmas[timestep]
            # print(self.alphas[timestep].size(), self.sigmas[timestep].size(), (eps_t - eps).size())
            grad_z_factor = self.alphas[timestep]**0.5 * self.sigmas[timestep] 
            grad_z = grad_z_factor.view(-1,1,1,1) * (eps_t - eps)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach().float(), 0.0, 0.0, 0.0)

        sds_loss = grad_z.clone() * latent_z
        del grad_z

        sds_loss = sds_loss.sum(1).mean()
        return sds_loss

class ToneLoss(nn.Module):
    def __init__(self, cfg):
        super(ToneLoss, self).__init__()
        self.dist_loss_weight = cfg.loss.tone.dist_loss_weight
        self.im_init = None
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.blurrer = torchvision.transforms.GaussianBlur(kernel_size=(cfg.loss.tone.pixel_dist_kernel_blur,
                                                                        cfg.loss.tone.pixel_dist_kernel_blur), sigma=(cfg.loss.tone.pixel_dist_sigma))

    def set_image_init(self, im_init):
        self.im_init = im_init.permute(2, 0, 1).unsqueeze(0)
        self.init_blurred = self.blurrer(self.im_init)

        # print(self.init_blurred.size())
        pil_image = self.init_blurred.squeeze(0)
        # print(pil_image.size())
        pil_image = torchvision.transforms.ToPILImage()(pil_image)
        pil_image.save("tmp.png")

    def get_scheduler(self, step=None, peak=[300,700,900]):
        if step is not None:
            schedule = 0
            for p in peak:
                schedule += self.dist_loss_weight * np.exp(-(1/5)*((step-p)/(20)) ** 2)

            return schedule
        else:
            return self.dist_loss_weight

    def forward(self, cur_raster, step=None):
        blurred_cur = self.blurrer(cur_raster)
        # print(self.init_blurred.size())
        pil_image = blurred_cur.squeeze(0)
        # print(pil_image.size())
        pil_image = torchvision.transforms.ToPILImage()(pil_image)
        pil_image.save("tmp2.png")
        return self.mse_loss(self.init_blurred.detach(), blurred_cur) * self.get_scheduler(step)
            
class DistanceTransformLoss:
    # インスタンス生成時にshape_img(最終の形状の画像)を入力
    # インスタンス呼出時にimg(最適化したいrendered画像)を入力
    def __init__(self, shape_img, ) :
        self.shape_img = shape_img
        self.dist = self.make_dist()
        content = self.shape_to_content(shape_img)
        self.target_dist = self.calc_dist(content, self.dist)

    def shape_to_content(self, shape_img, gamma=1.0): 
        content_image = np.array(shape_img)
        content_image = torch.from_numpy(content_image).to(torch.float32) / 255.0
        content_image = content_image.pow(gamma)
        content_image = content_image.to(pydiffvg.get_device())
        content_image = content_image.unsqueeze(0)
        content_image = content_image.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return content_image

    def make_dist(self):
        # self.shape_imgからdist_mapを作成する
        np_img = np.array(self.shape_img, dtype=np.uint8)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        dist = cv2.distanceTransform(gray, cv2.DIST_L2, maskSize=0)
        cv2.normalize(dist, dist, 0, 100.0, cv2.NORM_MINMAX)
        dist = torch.from_numpy(dist).to(torch.float32)
        dist = dist.pow(1.0)
        dist = dist.to(pydiffvg.get_device())
        return dist
    
    def calc_dist(self, img, dist):
        # self_distとimgのピクセルごとの距離を計る
        target_dist = img.clone()
        target_dist = 255 - target_dist
        for i in range(3):
            target_dist[:, i, :, :] = target_dist[:, i, :, :] * dist
        return target_dist
    
    def __call__(self, img):
        # return the Distance Transform Loss
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
        img_dist = self.calc_dist(img, self.dist)
        return (img_dist - self.target_dist).pow(2).mean()

class ClipLoss:
    def __init__(self, cfg, model="ViT-B/16"):
        self.caption = cfg.caption # テキストエンコーダ入力
        self.device = pydiffvg.get_device()
        self.model, self.preprocess = clip.load(model,device=self.device)

        with torch.no_grad():
            self.tokens = clip.tokenize(self.caption).to(self.device)
            text_sourse = self.model.encode_text(self.tokens).detach()
            self.text_features = text_sourse.mean(axis=0, keepdim=True)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def clip_normalize(self, image):
        image = nnf.interpolate(image, size=224, mode="bicubic")
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(self.device)
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)

        image = (image - mean) / std
        return image

    def __call__(self, img):
        # img: rendered image
        # print("CLIP", pydiffvg.get_device())
        # print(img.size())
        img_features = self.model.encode_image(self.clip_normalize(img))
        img_features /= img_features.clone().norm(dim=-1, keepdim=True)
        # print(img_features.size(), self.text_features)
        loss = 0
        loss -= torch.cosine_similarity(self.text_features, img_features, dim=1)
        # print(loss)
        loss = torch.sum(loss)
        return loss






class ConformalLoss:
    def __init__(self, parameters: EasyDict, device: torch.device, target_letter: str, shape_groups):
        self.parameters = parameters
        self.target_letter = target_letter
        self.shape_groups = shape_groups
        self.faces = self.init_faces(device)
        self.faces_roll_a = [torch.roll(self.faces[i], 1, 1) for i in range(len(self.faces))]

        with torch.no_grad():
            self.angles = []
            self.reset()


    def get_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []
        for i in range(len(self.faces)):
            triangles = points[self.faces[i]]
            triangles_roll_a = points[self.faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_
    
    def get_letter_inds(self, letter_to_insert):
        for group, l in zip(self.shape_groups, self.target_letter):
            if l == letter_to_insert:
                letter_inds = group.shape_ids
                return letter_inds[0], letter_inds[-1], len(letter_inds)

    def reset(self):
        points = torch.cat([point.clone().detach() for point in self.parameters.point])
        self.angles = self.get_angles(points)

    def init_faces(self, device: torch.device) -> torch.tensor:
        faces_ = []
        for j, c in enumerate(self.target_letter):
            points_np = [self.parameters.point[i].clone().detach().cpu().numpy() for i in range(len(self.parameters.point))]
            start_ind, end_ind, shapes_per_letter = self.get_letter_inds(c)
            print(c, start_ind, end_ind)
            holes = []
            if shapes_per_letter > 1:
                holes = points_np[start_ind+1:end_ind]
            poly = Polygon(points_np[start_ind], holes=holes)
            poly = poly.buffer(0)
            points_np = np.concatenate(points_np)
            faces = Delaunay(points_np).simplices
            is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces], dtype=bool)
            faces_.append(torch.from_numpy(faces[is_intersect]).to(device, dtype=torch.int64))
        return faces_

    def __call__(self) -> torch.Tensor:
        loss_angles = 0
        points = torch.cat(self.parameters.point)
        angles = self.get_angles(points)
        for i in range(len(self.faces)):
            loss_angles += (nnf.mse_loss(angles[i], self.angles[i]))
        return loss_angles

class UntieLoss:
    def __init__(self, shapes, shape_groups, ):
        """ ベジェの凸包性+分割によって、shape_groupsに含まれるすべてのパスに関して凸包な部分の重なりあう領域を少なくするようにロスを出力する。"""
        self.shapes = shapes
        self.shape_groups = shape_groups
        self.device = pydiffvg.get_device()

    def get_all_paths(self, shape_group):
        """ shape_groupに含まれるshapesを構成する全てのpath制御点をtensor[n, 4]で出力する"""
        paths = torch.empty((0,4,2),device =self.device)
        shape_ids = shape_group.shape_ids
        for i in shape_ids:
            shape = self.shapes[i]
            ncps = shape.num_control_points
            ps = shape.points
            for j in range(len(ncps)):
                path = ps[j*3:j*3+2]
                if j!=len(ncps)-1:
                    path = torch.cat((path,path[(j+1)*3]),dim=0)
                else:
                    path = torch.cat((path,path[0]),dim=0)
                paths = torch.cat((paths,path.unsqueeze(0)),dim=0)
        return paths

    def judge_overlap_paths(self, paths):
        """ pathsに含まれるすべてのpathに対して、2つのpath（同じパス同士を含む）の凸包矩形領域が重なるかどうかを判定 """
        pass

    @staticmethod
    def _judge_overlap(path1,path2):
        # 下請け。2つのpathに関して矩形領域が重なるかどうか
        # pathは[4,2]の4つの点群データ。
        pass
    
    @staticmethod
    def de_casteljau(t, coefs):
        beta = [c for c in coefs] # values in this list are overridden
        n = len(beta)
        for j in range(1, n):
            for k in range(n - j):
                beta[k] = beta[k] * (1 - t) + beta[k + 1] * t
        return beta[0]

class ToneFillLoss(nn.Module):
    def __init__(self, cfg):
        super(ToneFillLoss, self).__init__()
        self.dist_loss_weight = cfg.loss.tone.dist_loss_weight
        self.im_init = None
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.blurrer = torchvision.transforms.GaussianBlur(
            kernel_size=(cfg.loss.tone.pixel_dist_kernel_blur,
                         cfg.loss.tone.pixel_dist_kernel_blur),
            sigma=(cfg.loss.tone.pixel_dist_sigma))

    def set_image_init(self, im_init):
        self.im_init = im_init.permute(2, 0, 1).unsqueeze(0)
        self.init_blurred = self.blurrer(self.im_init)

    def get_scheduler(self, step=None):
        if step is not None:
            return self.dist_loss_weight * np.exp(-(1/5)*((step-300)/(20)) ** 2)
        else:
            return self.dist_loss_weight

    def forward(self, cur_raster, step=None):
        blurred_cur = self.blurrer(cur_raster)
        return self.mse_loss(self.init_blurred.detach(), blurred_cur) * self.get_scheduler(step)
            
class CircumferenceLoss:
    def __init__(self, shapes, shape_groups):
        self.shapes = shapes
        self.shape_groups = shape_groups

    def calc_loss(shapes):
        # 全てのshapesが閉じたパスだと仮定して、凸包領域内のいずれかの場所を中心する。
        # 中心からshapes.pointsの座標への方向ベクトルを使って、shapes.pointsの各点が前後の点との（角度的な）中間に存在するようにする。

        # 1. anker[n]がanker[n+1]を追い越す場合にロスが発生
        # 2．ctrl1, ctrl2はanker[n]とanker[n+1]の間に存在すると仮定。範囲外に存在する場合はその角度によってロスが発生。
        # 2another. ctrls
        # 3. ctrl1, ctrl2の順番が逆転した場合には追い越し角度とctrl1, ctrl2の距離によってロスを発生する。
        loss = 0.0
        for shape in shapes:
            points = shape.points
            # CircumferenceLoss._is_order(points)
            cx,cy = CircumferenceLoss.find_center(points)
            # cx = torch.tensor(1.0)
            # cy = torch.tensor(41.0)
            center = torch.stack([cx,cy])
            # self.check_ankers_in_order(points, center)
            # self.check_ctrls_between_ankers(points, center)
            l = CircumferenceLoss.check_all_points_in_order(points, center)
            loss = loss + l
        loss = loss/len(shapes)
        # print(loss)
        return loss

    def check_xtype_clossing(self, points, center):
        # x型の交差が存在する（可能性）をチェックする
        # 今回はセグメント同士が絶対に交差しない(anker+ctrlが完全に順巡り)であると仮定している
        # ↑の条件によりv型のセグメントが外部で交差する可能性がないため、x型の交差で輪っかができない = セグメントの点を用いてベジェ曲線の媒介変数表示をした時の3次式がx,yの場合にも判別式D<=0であればいい
        # つまり、x,yの両方とも判別式D>0ならば交差する
        # ただし 0<=t<=1
        for i in range(points.size()[0]/3):
            anker1 = points[i*3]
            ctrl1 = points[i*3+1]
            ctrl2 = points[i*3+2]
            anker2 = points[i*3+3] if i!=points.size()[0]/3-1 else points[0]
            
            # x型の場合だけ判定するべき。他の型の場合はDで判定した場合でも交差しない。
            angle1 = self._calc_angle(ctrl1-center, anker1-center)
            angle2 = self._calc_angle(ctrl2-center, anker1-center)
            

            a = 3*(-anker1 + 3*ctrl1 - 3*ctrl2 + anker2)
            b = 3*anker1 - 6*ctrl1 + 3*ctrl2
            c = -3*anker1 + 3*ctrl1
            D = b**2 - a*c
            ns = torch.zeros((2),dtype=int)
            i = 0
            for d in D:
                if d > 0:
                    ans1 = (-b-torch.sqrt(d))/a
                    ans2 = (-b+torch.sqrt(d))/a
                    if 0<=ans1 and ans1<=1:
                        ns[i] += 1
                    if 0<=ans2 and ans2<=1:
                        ns[i] += 1
                i += 1
        return ns

    def check_ctrls_between_ankers(self, points, center):
        # centerを中心として各ctrlsが前後のアンカーの(角度的な）間に存在するかどうかを確かめる
        # ただし、対応するctrls自体は別に逆になっていても良い
        losses = torch.zeros((points.size()[0]))
        for i in range(points.size()[0]/3):
            anker1 = points[i*3]
            ctrl1 = points[i*3+1]
            ctrl2 = points[i*3+2]
            anker2 = points[i*3+3] if i!=points.size()[0]/3-1 else points[0]

            region = self._calc_angle(anker2-center, anker1-center)
            angle1 = self._calc_angle(ctrl1-center, anker1-center)
            angle2 = self._calc_angle(ctrl2-center, anker1-center)
            if torch.sign(region)*torch.sign(angle1) >= 0:
                # 回転方向が同じ
                a = region-angle1
                if a < 0:
                    # ctrl1がanker2を追い越した場合
                    losses[i*3+1] = a
            else:
                # 回転方向が逆 = anker1より前にctrl1
                losses[i*3+1] = a
            if torch.sign(region)*torch.sign(angle2) >= 0:
                a = region-angle2
                if a < 0:
                    losses[i*3+2] = a
            else:
                losses[i*3+2] = a
        print("dasdadas", losses)
        return losses

    def check_ankers_in_order(self, points, center):
        # cx,cyを中心として各アンカーが前後のアンカーの（角度的に）間にあるかどうかを確かめる。
        # points: all (ankers + controls) points
        losses = torch.zeros((points.size()[0]/3))
        pre_anker = points[-3] # lust anker 
        # anker_order = torch.zeros((points.size()[0]/3), dtype=float)
        for  i in range(points.size()[0]/3):
            cur_anker = points[i*3]
            next_anker = points[i*3+3] if i!=points.size()[0]/3-1 else points[0]
            cur_angle = self._calc_angle(cur_anker-center, pre_anker-center)
            next_angle = self._calc_angle(next_anker-center, pre_anker-center)

            a = next_angle - cur_angle
            if torch.sign(cur_angle)*torch.sign(next_angle) > 0:
                # 回転方向が同じ場合(反時計回りしか考察していないが)
                if a < 0:
                    # cur_ankerがnext_ankerを追い越してしまう場合
                    losses[i*3] = a
            else :
                # 回転方向が逆の場合
                # cur_ankerがpre_ankerよりも前にいる場合
                losses[i*3] = a
            pre_anker = cur_anker
        
        print("self.losses: ", losses, losses.requires_grad, losses.is_leaf,)
        return losses

    def check_all_points_in_order(points, center):
        # 全てのポイント版（簡単バージョン）
        # 全部の点が中心に対して順巡りなら交差するわけないよねっていう。
        # losses = torch.zeros((points.size()[0]))
        loss = 0.0
        n = points.size()[0]
        centers = center.repeat((n,1))
        pre_points = torch.cat((points[-1].unsqueeze(0),points[:-1]),dim=0)
        # next_points = torch.cat((points[1:],points[0].unsqueeze(0)),dim=0)
        cur_angles = CircumferenceLoss._calc_angles(pre_points-centers, points-centers)
        # next_angles = CircumferenceLoss._calc_angle(pre_points-centers, next_point-centers)
        cur_angles = cur_angles * -1
        cur_angles = torch.nn.functional.relu(cur_angles)
        loss = torch.sum(cur_angles)
        return loss



        # pre_point = points[-1] # last point 
        # for i in range(points.size()[0]):
        #     cur_point = points[i]
        #     next_point = points[i+1] if i<points.size()[0]-1 else points[0]
        #     cur_angle = CircumferenceLoss._calc_angle(pre_point-center, cur_point-center)
        #     next_angle = CircumferenceLoss._calc_angle(pre_point-center, next_point-center)
        #     # print(cur_angle/torch.pi*180, next_angle/torch.pi*180)

        #     if cur_angle<0:
        #         loss = loss + -1 * cur_angle
        #     # elif cur_angle>next_angle:
        #     #     loss = loss + (cur_angle-next_angle)
        #     pre_point = points[i]
        # # loss = loss/points.size()[0]  
        # # print(loss)
        return loss

    def _calc_angles(p1,p2):
        # _cala_angleの一括処理版
        dot_product = torch.sum(p1*p2, dim=1) # 内積
        sign = torch.sign(p1[:,0]*p2[:,1] - p1[:,1]*p2[:,0])
        norm_v1 = torch.norm(p1,dim=1)
        norm_v2 = torch.norm(p2,dim=1)
        cosine_theta = dot_product / (norm_v1 * norm_v2)
        cosine_theta = torch.clamp(cosine_theta, -1.0+1e-7, 1.0-1e-7)
        theta_rad = torch.acos(cosine_theta) # 原因こいつ epsの追加で解
        return sign * theta_rad

    @staticmethod
    def _calc_angle(v1, v2):
        # 回転方向(反時計回り+)を考慮した角度(v1が基準)
        # -pi ~ piで出力
        dot_product = torch.dot(v1, v2)
        norm_v1 = torch.norm(v1)
        norm_v2 = torch.norm(v2)
        
        cosine_theta = dot_product / (norm_v1 * norm_v2)
        cosine_theta = torch.clamp(cosine_theta, -1.0+1e-7, 1.0-1e-7)
        theta_rad = torch.acos(cosine_theta) # 原因こいつ epsの追加で解決
        if torch.isnan(theta_rad):
            print("koitu")
        
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        return torch.sign(cross_product) * theta_rad

    @staticmethod
    def find_center(points):
        # ankerの中央値を中心とする
        ankers = torch.empty((0,2))
        for i in range(int(points.size()[0]/3)):
            ankers = torch.cat((ankers,points[i*3].unsqueeze(0)),dim=0)
        n = int(ankers.size()[0])
        s, _ = torch.sort(ankers, dim=0)
        s = s.detach()
        if n%2 == 0 :
            cx = (s[int(n/2-1),0] + s[int(n/2),0]) / 2 
            cy = (s[int(n/2-1),1] + s[int(n/2),1]) / 2
        else:
            cx = s[int(n/2),0]
            cy = s[int(n/2),1]
        return (cx,cy)

    def _is_order(points):
        # 点群が一定の回転方向を向いているかどうか。
        n = points.size()[0]
        
        # signs = torch.zeros(n, dtype=torch.float, device=diffvg.get_device())

        A = points
        B= torch.cat((points[1:],points[0].unsqueeze(0)),dim=0)
        C= torch.cat((points[2:],points[0:2]),dim=0)

        cross_product = (B[:,0] - A[:,0]) * (C[:,1] - B[:,1]) - (B[:,1] - A[:,1]) * (C[:,0] - B[:,0])
        signs = torch.sign(cross_product)
        print("dsadsa",signs.size(),cross_product.size(),signs.requires_grad,signs.is_leaf)
        print(signs)

        return 
