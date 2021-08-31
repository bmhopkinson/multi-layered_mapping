import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from Camera import Frame, Camera
from aabbtree_mod import AABB, AABBTree
import json


def load_mesh():
    mesh = trimesh.load_mesh('./data/mesh_data/Sapelo_202106_run15/mesh.ply')
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
    vertices = mesh.vertices.view(np.ndarray)
    faces = mesh.faces.view(np.ndarray)

    cameras, frames = load_agisoft_data()

    frame_test = frames[200]
    camera_test = cameras[frame_test.camera_id]
    tree = AABBTree()
    tree = tree.from_mesh_faces(mesh)

    hits = frame_test.project_from_tree(tree)


    mesh.visual.face_colors[hits] = np.array([255, 0, 0, 125], dtype = np.uint8)
    mesh.show()
    # box = trimesh.primitives.Box(
    #             center=[0, 0, 0],
    #             extents=[1, 1, 1])
    #
    # combo = mesh + box
    # combo.show()
    print('done')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
