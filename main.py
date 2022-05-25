from core.train_agent import TrainAgent
from core.hotspot_agent import HotSpotAgent

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

    if "strategy" in config and config["strategy"] == "hotspot":
        agent = HotSpotAgent(config)
    else:
        agent = TrainAgent(config)
    agent.train()
        
    