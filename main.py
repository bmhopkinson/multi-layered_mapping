import numpy as np
import trimesh
import xml
import xml.etree.ElementTree as ET
from Frame import Frame, Camera

def load_mesh():
    mesh = trimesh.load_mesh('./data/mesh_data/Sapelo_202106_run13/mesh.ply')
    return mesh

    #TODO: load Agisoft cameras, figure out AABB, start projecting


def load_agisoft_data():
    tree = ET.parse('./data/mesh_data/Sapelo_202106_run13/agisoft_cameras_Imaging.xml')
    root = tree.getroot()
    version = root.attrib['version']
    print(version)

    chunks = root.findall('chunk')
    cameras = {}
    frames = []

    for chunk in chunks:
        cameras_this_chunk = chunk.find('sensors')  #my terminolgy 'camera' = agisoft 'sensor'
        for camera in cameras_this_chunk:
            cam = Camera()
            cam.load_agisoft(camera, version)
            cameras[cam.id] = cam

        frames_this_chunk = chunk.find('cameras') #my terminolgy 'frame' = agisoft 'camera'
        for frame in frames_this_chunk:
            _frame = Frame()
            _frame.load_agisoft(frame, cameras)
            frames.append(_frame)


    return cameras, frames


if __name__ == '__main__':
    mesh = load_mesh()
    cameras, frames = load_agisoft_data()

    frame_test = frames[300]
    camera_test = cameras[frame_test.camera_id]

    vertices = mesh.vertices.view(np.ndarray)
    hits = []
    for vertex in vertices:
        vertex = np.append(vertex, 1.000) #make homogeneous
        vertex = vertex.reshape((4,1))

        valid, x_cam = frame_test.project(vertex)
        if valid:
            hits.append({
                'vertex': vertex,
                'x': x_cam
            })
    print('done')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
