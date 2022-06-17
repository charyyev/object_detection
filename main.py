from core.train_agent import TrainAgent
from core.hotspot_agent import HotSpotAgent
from core.aux_agent import AuxAgent

import argparse
import json



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--config', type=str, nargs='?', default = "/home/stpc/proj/object_detection/configs/base.json")
    # Parse the argument
    args = parser.parse_args()


    f = open(args.config)
    config = json.load(f)

    if "strategy" in config:
        if config["strategy"] == "hotspot":
            agent = HotSpotAgent(config)
        elif config["strategy"] == "aux":
            agent = AuxAgent(config)
    else:
        agent = TrainAgent(config)
    agent.train()
        
    