import numpy as np
import trimesh
import xml
import xml.etree.ElementTree as ET
from Frame import Frame, Camera

def load_mesh():
    mesh = trimesh.load_mesh('./data/mesh_data/Sapelo_202106_run13/mesh.ply')
    return mesh

    #TODO: load Agisoft cameras, figure out AABB, start projecting

def parse_camera_data(camera, version):
    camera_data = {}
    # find id
    camera_data['id'] = camera.attrib['id']

    # extract image dimensions
    dim = camera.find('resolution')
    camera_data['dim'] = [float(dim.attrib['width']), float(dim.attrib['height'])]

    # process calibration, distortion parameters are present in variable numbers
    calib = camera.find('calibration')
    K = np.eye(3, dtype=float)

    fx = []
    fy = []
    if version == '1.4.0':
        fx = float(calib.find('f').text)
        fy = fx
    else:
        fx = float(calib.find('fx').text)
        fy = float(calib.find('fy').text)

    K[0, 0] = fx
    K[1, 1] = fy
    camera_data['fx'] = fx
    camera_data['fy'] = fy

    cx = float(calib.find('cx').text)
    cy = float(calib.find('cy').text)
    if version == '1.4.0':
        cx = cx + 0.5 * camera_data['dim'][0]
        cy = cy + 0.5 * camera_data['dim'][1]

    K[0, 2] = cx
    K[1, 2] = cy
    camera_data['cx'] = cx
    camera_data['cy'] = cy
    camera_data['K']  = K

    # process distortion parameters, first radial distorion parameters (ks), then tangential (ps)
    k1 = calib.find('k1')
    if not k1 is None:
        camera_data['k1'] = float(k1.text)
    else:
        camera_data['k1'] = 0.000

    k2 = calib.find('k2')
    if not k2 is None:
        camera_data['k2'] = float(k2.text)
    else:
        camera_data['k2'] = 0.000

    k3 = calib.find('k3')
    if not k3 is None:
        camera_data['k3'] = float(k3.text)
    else:
        camera_data['k3'] = 0.000

    p1 = calib.find('p1')
    if not p1 is None:
        camera_data['p1'] = float(p1.text)
    else:
        camera_data['p1'] = 0.000

    p2 = calib.find('p2')
    if not p2 is None:
        camera_data['p2'] = float(p2.text)
    else:
        camera_data['p2'] = 0.000

    return camera_data

def parse_frame_data(frame, cameras, version):
    frame_data = {}
    frame_data['frame_id'] = frame.attrib['id']
    frame_data['label'] = frame.attrib['label']
    frame_data['camera_id'] = frame.attrib['sensor_id']
    frame_data['enabled'] = frame.attrib['enabled']

    transform = frame.find('transform')
    tdata_raw = [float(elm) for elm in transform.text.split()]
    frame_data['Twc'] = np.array(tdata_raw).reshape((4,4))   #camera to world transform
    frame_data['Tcw'] = np.linalg.inv(frame_data['Twc'])        # world to camera transform
    K = cameras[frame_data['camera_id']]['K']
    frame_data['P'] = np.matmul(K, frame_data['Tcw'][0:3,:])  #projection matrix
    return frame_data
    #print('split')




def load_agisoft_data():
    tree = ET.parse('./data/mesh_data/Sapelo_202106_run13/agisoft_cameras_Imaging.xml')
    root = tree.getroot()
    version = root.attrib['version']
    print(version)

    chunks = root.findall('chunk')
    cameras = {}
    cameras_alt = {}
    frames = []
    frames_alt = []

    for chunk in chunks:
        cameras_this_chunk = chunk.find('sensors')  #my terminolgy 'camera' = agisoft 'sensor'
        for camera in cameras_this_chunk:
            cam_alt = Camera()
            cam_alt.load_agisoft(camera, version)
            cameras_alt[cam_alt.id] = cam_alt

            camera_data = parse_camera_data(camera, version)  #open dictionary
            cameras[camera_data['id']] = camera_data

        frames_this_chunk = chunk.find('cameras') #my terminolgy 'frame' = agisoft 'camera'
        for frame in frames_this_chunk:
            frame_alt = Frame()
            frame_alt.load_agisoft(frame , cameras_alt)
            frames_alt.append(frame_alt)
            frames.append(parse_frame_data(frame, cameras, version))

    return cameras, frames, frames_alt


if __name__ == '__main__':
    mesh = load_mesh()
    cameras, frames, frames_alt = load_agisoft_data()

    frame_test = frames[300]
    frame_alt_test = frames_alt[300]
    camera_test = cameras[frame_test['camera_id']]

    vertices = mesh.vertices.view(np.ndarray)
    hits = []
    hits_alt =[]
    for vertex in vertices:
        vertex = np.append(vertex, 1.000) #make homogeneous
        vertex = vertex.reshape((4,1))

        valid, x_cam = frame_alt_test.project(vertex)
        if valid:
            hits_alt.append({
                'vertex': vertex,
                'x': x_cam
            })

        xh = np.matmul(frame_test['P'], vertex)
        if(xh[2] > 0 ):  #point must be in front of camera
            x = [xh[0]/xh[2], xh[1]/xh[2]]
            #is x within image
            if( (0 < x[0] < camera_test['dim'][0]) and (0 < x[1] < camera_test['dim'][1])):

                hits.append({
                            'vertex': vertex,
                            'x': x
                            })
    print('done')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
