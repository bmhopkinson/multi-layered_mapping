import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from Camera import Frame, Camera
from aabbtree_mod import AABB, AABBTree

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
    faces = mesh.faces.view(np.ndarray)

    face_lb = vertices[faces[:, 0], :]
    face_ub = vertices[faces[:, 0], :]

    for i in range(2):
        face_lb = np.minimum(face_lb, vertices[faces[:, i + 1], :])
        face_ub = np.maximum(face_ub, vertices[faces[:, i + 1], :])

    hits = []

    aabb_init = []
    for lb,ub in zip(face_lb, face_ub):
        aabb_init.append(AABB( limits=[(lb[0], ub[0]), (lb[1], ub[1]), (lb[2], ub[2])] ))

    tree = AABBTree(aabbs_init=aabb_init)
    print('done')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
