import torch

from core.models.pixor_torchscript import PIXOR

if __name__ == "__main__":
    geometry = {
                "y_min": -40.0,
                "y_max": 40.0,
                "x_min": -35.0,
                "x_max": 35.0,
                "z_min": -2.5,
                "z_max": 1.0,
                "x_res": 0.1,
                "y_res": 0.1,
                "z_res": 0.1,
                "input_shape": [800, 700, 35],
                "label_shape": [200, 175, 6]
            }
    
    model_path = "/home/stpc/experiments/pixor_mixed_submap_21-04-2022_1/1319epoch"

    data_file = "/home/stpc/clean_data/list/test.txt"
    
    model = PIXOR()
    #model.to(config['device'])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    #device = config["device"]
    model.half()
    scripted_model = torch.jit.script(model)

    print(scripted_model.code)
    scripted_model.save("/home/stpc/models/pixor_half.pt")