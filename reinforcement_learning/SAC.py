"""
file: ppo_train.py
author: Tawn Kramer
date: 13 October 2018
notes: ppo2 test from stable-baselines here:
https://github.com/hill-a/stable-baselines
"""
import argparse
import uuid

import gym
from stable_baselines3 import SAC,TD3
import tensorflow as tf
import gym_donkeycar
if __name__ == "__main__":
    # Initialize the donkey environment
    # where env_name one of:
    env_list = [
        "donkey-warehouse-v0",
        "donkey-generated-roads-v0",
        "donkey-avc-sparkfun-v0",
        "donkey-generated-track-v0",
        "donkey-roboracingleague-track-v0",
        "donkey-waveshare-v0",
        "donkey-minimonaco-track-v0",
        "donkey-warren-track-v0",
        "donkey-thunderhill-track-v0",
        "donkey-circuit-launch-track-v0",
        "donkey-imt-track-v0"
    ]

    parser = argparse.ArgumentParser(description="ppo_train")
    parser.add_argument(
        "--sim",
        type=str,
        default="sim_path",
        help="path to unity simulator. maybe be left at manual if you would like to start the sim on your own.",
    )
    parser.add_argument("--port", type=int, default=9091, help="port to use for tcp")
    parser.add_argument("--test", action="store_true", help="load the trained model and play")
    parser.add_argument("--multi", action="store_true", help="start multiple sims at once")
    parser.add_argument(
        "--env_name", type=str, default="donkey-imt-track-v0", help="name of donkey sim environment", choices=env_list
    )

    args = parser.parse_args()
    
    
    if args.sim == "sim_path" and args.multi:
        print("you must supply the sim path with --sim when running multiple environments")
        exit(1)

    env_id = args.env_name


    conf = {
        "exe_path": args.sim,
        "host": "127.0.0.1",
        "port": args.port,
        "body_style": "donkey",
        "body_rgb": (128, 128, 128),
        "car_name": "IMT",
        "font_size": 100,
        "racer_name": "DDQN",
        "country": "USA",
        "bio": "Learning to drive w DDQN RL",
        "guid": str(uuid.uuid4()),
        "max_cte": 10,
    }



        # make gym env
    env = gym.make(args.env_name, conf=conf)
    model = SAC("CnnPolicy", env, verbose=1, buffer_size=1000)
    model.learn(total_timesteps=10000)

    obs, info = env.reset()
    i=0
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
        if i % 100 == 0:
            print("saving...")
            model.save("SAC_donkey")
        i+=1

    env.close()
