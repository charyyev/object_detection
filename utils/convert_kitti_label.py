import os
import numpy as np

def read_data(label_folder, calib_folder):
    label_files = sorted(os.listdir(label_folder))
    label_paths = []
    calib_paths = []
    for label_file in label_files:
        label_path = os.path.join(label_folder, label_file)
        calib_path = os.path.join(calib_folder, label_file)
        calib_paths.append(calib_path)
        label_paths.append(label_path)
    
    return label_paths, 
    
def read_calib(calib_path):
    lines = open(calib_path).readlines()
    lines = [ line.split()[1:] for line in lines ][:-1]

    Tr_velo_to_cam = np.array(lines[5]).reshape(3,4)
    Tr_velo_to_cam = np.concatenate(  [ Tr_velo_to_cam, np.array([0,0,0,1]).reshape(1,4)  ]  , 0     )

    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')

    return Tr_velo_to_cam

if __name__ == "__main__":
    label_folder = "/home/stpc/data/kitti/label_2/training/label_2"
    calib_folder = "/home/stpc/data/kitti/calib/training/calib"
    save_folder = "/home/stpc/data/kitti/label_2/training/label_2_reduced"

    #label_paths, calib_paths = read_data(label_folder, calib_folder)

    label_files = sorted(os.listdir(label_folder))
    for label_file in label_files:
        label_path = os.path.join(label_folder, label_file)
        calib_path = os.path.join(calib_folder, label_file)
        reduced_label_path = os.path.join(save_folder, label_file)

        object_list = {'Car': 1, 'Pedestrian':2, 'Person_sitting':2, 'Cyclist':3}
        velo_to_cam = read_calib(calib_path)
        corner_list = []
        class_list = []
        new_entries = []
        with open(label_path, 'r') as f:
            lines = f.readlines() # get rid of \n symbol
            for line in lines:
                bbox = []
                entry = line.split(' ')
                name = entry[0]
                if name in list(object_list.keys()):
                    new_entry = name
                    cam_to_velo = np.linalg.inv(velo_to_cam)

                    vals = [float(e) for e in entry[8:15]]
                    h, w, l, x, y, z, yaw = vals
                    yaw = -(yaw + np.pi / 2)

                    velo_coord = np.matmul(cam_to_velo, np.array([[x], [y], [z], [1]]))
                    x = velo_coord[0][0]
                    y = velo_coord[1][0]
                    z = velo_coord[2][0]

                    new_entry += " " + str(h) + " " + str(w) + " " + str(l) + " " + str(x) + " " + str(y) + " " + str(z) + " " + str(yaw)
                    new_entries.append(new_entry)
        
        with open(reduced_label_path, 'w') as f:
            for new_entry in new_entries:
                f.write(new_entry)
                f.write('\n')





