import numpy as np
import trimesh
import xml
import xml.etree.ElementTree as ET
from Camera import Camera, Sensor

def load_mesh():
    mesh = trimesh.load_mesh('./data/mesh_data/Sapelo_202106_run13/mesh.ply')
    return mesh

    #TODO: load Agisoft cameras, figure out AABB, start projecting

def parse_sensor_data(sensor, version):
    sensor_data = {}
    # find id
    sensor_data['id'] = sensor.attrib['id']

    # extract image dimensions
    dim = sensor.find('resolution')
    sensor_data['dim'] = [float(dim.attrib['width']), float(dim.attrib['height'])]

    # process calibration, distortion parameters are present in variable numbers
    calib = sensor.find('calibration')
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
    sensor_data['fx'] = fx
    sensor_data['fy'] = fy

    cx = float(calib.find('cx').text)
    cy = float(calib.find('cy').text)
    if version == '1.4.0':
        cx = cx + 0.5 * sensor_data['dim'][0]
        cy = cy + 0.5 * sensor_data['dim'][1]

    K[0, 2] = cx
    K[1, 2] = cy
    sensor_data['cx'] = cx
    sensor_data['cy'] = cy
    sensor_data['K']  = K

    # process distortion parameters, first radial distorion parameters (ks), then tangential (ps)
    k1 = calib.find('k1')
    if not k1 is None:
        sensor_data['k1'] = float(k1.text)
    else:
        sensor_data['k1'] = 0.000

    k2 = calib.find('k2')
    if not k2 is None:
        sensor_data['k2'] = float(k2.text)
    else:
        sensor_data['k2'] = 0.000

    k3 = calib.find('k3')
    if not k3 is None:
        sensor_data['k3'] = float(k3.text)
    else:
        sensor_data['k3'] = 0.000

    p1 = calib.find('p1')
    if not p1 is None:
        sensor_data['p1'] = float(p1.text)
    else:
        sensor_data['p1'] = 0.000

    p2 = calib.find('p2')
    if not p2 is None:
        sensor_data['p2'] = float(p2.text)
    else:
        sensor_data['p2'] = 0.000

    return sensor_data

def parse_camera_data(camera, sensors, version):
    camera_data = {}
    camera_data['camera_id'] = camera.attrib['id']
    camera_data['label'] = camera.attrib['label']
    camera_data['sensor_id'] = camera.attrib['sensor_id']
    camera_data['enabled'] = camera.attrib['enabled']

    transform = camera.find('transform')
    tdata_raw = [float(elm) for elm in transform.text.split()]
    camera_data['Twc'] = np.array(tdata_raw).reshape((4,4))   #camera to world transform
    camera_data['Tcw'] = np.linalg.inv(camera_data['Twc'])        # world to camera transform
    K = sensors[camera_data['sensor_id']]['K']
    camera_data['P'] = np.matmul(K, camera_data['Tcw'][0:3,:])  #projection matrix
    return camera_data
    #print('split')




def load_agisoft_cameras():
    tree = ET.parse('./data/mesh_data/Sapelo_202106_run13/agisoft_cameras_Imaging.xml')
    root = tree.getroot()
    version = root.attrib['version']
    print(version)

    chunks = root.findall('chunk')
    sensors = {}
    sensors_alt = {}
    cameras = []
    cameras_alt = []

    for chunk in chunks:
        sensors_this_chunk = chunk.find('sensors')
        for sensor in sensors_this_chunk:
            sens_alt = Sensor()
            sens_alt.load_agisoft(sensor, version)
            sensors_alt[sens_alt.id] = sens_alt

            sensor_data = parse_sensor_data(sensor, version)  #open dictionary
            sensors[sensor_data['id']] = sensor_data

        cameras_this_chunk = chunk.find('cameras')
        for camera in cameras_this_chunk:
            cam_alt = Camera()
            cam_alt.load_agisoft(camera, sensors_alt)
            cameras_alt.append(cam_alt)
            cameras.append(parse_camera_data(camera, sensors, version))

    return sensors, cameras, cameras_alt


if __name__ == '__main__':
    mesh = load_mesh()
    sensors, cameras, cameras_alt = load_agisoft_cameras()

    cam_test = cameras[300]
    cam_alt_test = cameras_alt[300]
    sensor_test = sensors[cam_test['sensor_id']]

    vertices = mesh.vertices.view(np.ndarray)
    hits = []
    hits_alt =[]
    for vertex in vertices:
        vertex = np.append(vertex, 1.000) #make homogeneous
        vertex = vertex.reshape((4,1))

        valid, x_cam = cam_alt_test.project(vertex)
        if valid:
            hits_alt.append({
                'vertex': vertex,
                'x': x_cam
            })

        xh = np.matmul(cam_test['P'], vertex)
        if(xh[2] > 0 ):  #point must be in front of camera
            x = [xh[0]/xh[2], xh[1]/xh[2]]
            #is x within image
            if( (0 < x[0] < sensor_test['dim'][0]) and (0 < x[1] < sensor_test['dim'][1])):

                hits.append({
                            'vertex': vertex,
                            'x': x
                            })
    print('done')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
