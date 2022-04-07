#from core.agents.train_agent import TrainAgent

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

    print(config)

    # agent = TrainAgent(config)
    # agent.train()
        
    