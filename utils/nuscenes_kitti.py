from nuscenes.nuscenes import NuScenes

if __name__ == "__main__":
    nusc = NuScenes(version='v1.0', dataroot='/home/stpc/data/v1.0-trainval_meta/v1.0-trainval', verbose=True)