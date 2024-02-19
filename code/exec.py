from main import main
import torch
from easydict import EasyDict as edict
import os
import random
import concurrent.futures

def main01(cfg):
    
    concepts = ["leaf", "Fire", "Rabbit", "Mt.Fuji", "Flowers", "Robot", "cherry blossoms", "smoke", "gears", "Tree", "Bear", "cat", "lightning", "stone", "heart", "Statue of Liberty"]
    targets = ["heart", "star", "test","water", "gear", "fire", "coffee", "LuckiestGuy-Regular_A_scaled","LuckiestGuy-Regular_R_scaled", "LuckiestGuy-Regular_S_scaled", "LuckiestGuy-Regular_T_scaled",]

    cfg.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    cfg.log_dir = "ex3/change_concepts"
    def task01(cfg, concepts, targets, device):
        print(device)
        cfg = edict(cfg)
        cfg.device = torch.device(device) if torch.cuda.is_available() else "cpu"
        for c in concepts:
            for shape_img in targets:
                cfg.semantic_concept = c
                cfg.caption = f"{cfg.prompt_prefix} {cfg.semantic_concept}. {cfg.prompt_suffix}"
                cfg.seed = random.randint(0,65535)
                f_name = f"{c}_{shape_img}"
                cfg.experiment_dir = os.path.join(cfg.log_dir, f_name)
                cfg.target = os.path.join('code/data/init', shape_img) 
                main(cfg)


    ngpu = torch.cuda.device_count()
    task_args = []  
    num_processes = ngpu

    print("Create a list of configuration objects")
    for n in range(ngpu):
        dvs = f"cuda:{n}"
        c_list = concepts[n * int(len(concepts)/ngpu):(n+1) * int(len(concepts)/ngpu)]
        if n == ngpu-1:
            c_list.append(concepts[(n+1) * int(len(concepts)/ngpu):])
        task_args.append((cfg.copy(), c_list, targets, dvs))

    print("execute task01")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        executor.map(lambda args: task01(*args), task_args)

if __name__ == "__main__":
    cfg = {
        'parent_config': 'baseline', 
        'save': {
            'init': True, 
            'image': True,
            'video': False, 
            'video_frame_freq': 1}, 
        'trainable': {
            'point': True}, 
        'lr_base': {
            'point': 1, 
            'color': 0.01}, 
        'lr': {
            'lr_init': 0.1, 
            'lr_final': 0.01,
            'lr_delay_mult': 0.1, 
            'lr_delay_steps': 100}, 
        'lr_color': {'lr_delay_mult': 0.1, 'lr_delay_steps': 50, 'lr_init': 1, 'lr_final': 1}, 

        'num_iter': 1000, 
        'render_size': 600, 
        'cut_size': 512, 
        'level_of_cc': 1, 
        'seed': 0, 
        'diffusion': {'model': 'runwayml/stable-diffusion-v1-5', 'timesteps': 1000, 'guidance_scale': 100}, 

        'loss': {
            'use_sds_loss': True, 
            'tone': {
                'use_tone_loss': True, 
                'dist_loss_weight': 200, 
                'pixel_dist_kernel_blur': 51, 
                'pixel_dist_sigma': 30}, 
            'conformal': {
                'use_conformal_loss': False, 
                'angeles_w': 0.5}, 
            'dt': {
                'use_dt_loss': False, 
                'dt_w': 10}, 
            'xing': {
                'use_xing_loss': False, 
                'xing_w': 100}}, 

        'num_paths': 200, 
        'bg': False, 
        'device': "cuda", 
        'crop_part': 0, 
        'config': 'code/config/base.yaml', 
        'experiment': 'svg_image', 

        'prompt_prefix': 'a logo of ',
        'semantic_concept': 'rabbit', 
        'prompt_suffix': 'minimal flat 2d vector. lineal color. trending on artstation',
        'caption': 'a logo of rabbit. minimal flat 2d vector. lineal color. trending on artstation', 

        'font': 'none', 
        'word': 'rabbit', 
        'optimized_letter': None, 
        'batch_size': 1, 
        'token': '', 

        'use_wandb': 0, 
        'wandb_user': 'none', 

        'letter': 'none_none_scaled', 
        'target': 'code/data/init/heart', 

        'experiment_dir': 'ex2/any2/svg_image_rabbit_1706799346/none/none_none_scaled_concept_rabbit_seed_46260',
        'log_dir': 'ex2/any2/svg_image_rabbit_1706799346', 
    }

    cfg = edict(cfg)
    main01(cfg)
