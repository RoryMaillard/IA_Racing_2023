import torch
import os
import gym
import torchvision.transforms as tf
import gym_donkeycar
import numpy as np
import importlib
import cv2


port = 9091

class Drive():
    def __init__(self,cfg):
        self.device = cfg['device']
        model_module=importlib.import_module(cfg['path_to_model_module'])
        self.model = getattr(model_module,cfg['model_name'])(cfg['model_creation_args'])
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(cfg['path_to_trained_model']))
        conf = { "exe_path" : cfg['sim_exe_path'], "port" : port }
        self.env = gym.make(cfg['env_track'], conf=conf)
        
    
    def image_to_tensor(self, cam_image_arr):
        transformImg=tf.Compose([ tf.ToPILImage(),
                             tf.ToTensor()]) 
        camera_image_tensor = transformImg(cam_image_arr) # Transform to pytorch
        camera_image_tensor = torch.unsqueeze(camera_image_tensor, 0)
        
        return camera_image_tensor
    
    def run(self, cam_image_arr):
        
        image_tensor = self.image_to_tensor(cam_image_arr)
        
        image_tensor = torch.autograd.Variable(image_tensor, requires_grad=False).to(self.device) # Load image
        
        outputs = self.model(image_tensor)
        
        throttle, angle = float(outputs[0][1]), float(outputs[0][0])
        
        return throttle, angle



# SET UP ENVIRONMENT
# You can also launch the simulator separately
# in that case, you don't need to pass a `conf` object


if __name__ == "__main__":
    cfg = {
        'device': "cuda",
        'path_to_model_module': 'model',
        'model_name':'MyFirstNet',
        'model_creation_args':'',
        'path_to_trained_model':'drive_models/423.torch',
        'sim_exe_path':"/home/rory/Documents/IMT/procom/DonkeySimLinux/donkey_sim.x86_64",
        'env_track':"donkey-waveshare-v0"
    }
    
    driver = Drive(cfg)
    obs = driver.env.reset()
    try:
        while True:
            action =  driver.run(cv2.cvtColor(obs,cv2.COLOR_BGR2RGB))# drive straight with small speed
            print(action)
            # execute the action
            obs, reward, done, info = driver.env.step(action)
    except KeyboardInterrupt:
    # You can kill the program using ctrl+c
        pass

        # Exit the scene
    driver.env.close()
    
# PLAY
