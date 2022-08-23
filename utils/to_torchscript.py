import torch

from core.models.pixor_torchscript import PIXOR
from core.models.mobilepixor_aux_torchscript import MobilePIXOR
from core.models.afdet_torchscript import AFDet

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
    
    #model_path = "/home/stpc/experiments/mobilepixor_more_classes_03-06-2022_1/174epoch"
    model_path = "/home/stpc/experiments/scconv/152epoch"
    data_file = "/home/stpc/clean_data/list/test.txt"
    model_type = "afdet"

    if model_type == "pixor":
        model = PIXOR()
    elif model_type == "mobilepixor":
        model = MobilePIXOR()
    else:
        model = AFDet()
    model.to("cuda:0")
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    #device = config["device"]
    model.half()
    scripted_model = torch.jit.script(model)

    print(scripted_model.code)
    scripted_model.save("/home/stpc/models/afdet.pt")