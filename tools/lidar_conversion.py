import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import rosbag


DATA_TYPE = {
  1: np.int8,
  2: np.uint8,
  3: np.int16,
  4: np.uint16,
  5: np.int32,
  6: np.uint32,
  7: np.float32,
  8: np.float64
}

DATA_SIZE = {
  1: 1,
  2: 1,
  3: 2,
  4: 2,
  5: 4,
  6: 4,
  7: 4,
  8: 8
}

DUMMY_FIELD_PREFIX = '__'


#input: pointcloud2 message
#return Nx4 numpy array

def pcl_to_numpy(msg):
    header = msg.header
    current_time = header.stamp.to_sec()

    #convert pointcloud to numpy structured array
    points = pointcloud2_to_array(msg)
    arr = np.zeros((len(points), 4) )
    
    arr[:, 0] = points['x']
    arr[:, 1] = points['y']
    arr[:, 2] = points['z']
    arr[:, 3] = points['intensity']
    
    return arr


def fields_to_dtype(fields, point_step):
    '''Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(
                ('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = DATA_TYPE[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += DATA_SIZE[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1

    return np_dtype_list


def pointcloud2_to_array(cloud_msg, squeeze=True):

    # construct a numpy record type equivalent to the point type of this cloud
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)

    # parse the cloud into an array
    cloud_arr = np.frombuffer(cloud_msg.data, dtype_list)

    # remove the dummy fields that were added
    cloud_arr = cloud_arr[
        [fname for fname, _type in dtype_list if not (
            fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]

    
    return np.reshape(cloud_arr, (cloud_msg.width * cloud_msg.height,))
        
   
            
            
