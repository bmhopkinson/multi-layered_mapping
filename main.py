import numpy as np
import cv2
import trimesh
import xml.etree.ElementTree as ET
from Camera import Frame, Camera
from MeshLabeler import MeshLabeler
from aabbtree import AABB, AABBTree
import json

""" script to label marsh mesh from images - uses semantically segmented images for class (plant) labeling and raw
     images for coloring mesh for visualizations """

mesh_file = './data/Sapelo_202106_run15/mesh_fine.ply'
camera_file = './data/Sapelo_202106_run15/agisoft_cameras_Imaging.xml'
image_folder = './data/Sapelo_202106_run15/imaging_preds_3by2/'
image_raw_folder ='./data/Sapelo_202106_run15/imaging/'


def load_mesh():
    mesh = trimesh.load_mesh(mesh_file)
    return mesh

def load_agisoft_data():
    tree = ET.parse(camera_file)
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

    tree = AABBTree()
    tree = tree.from_mesh_faces(mesh)

    cameras, frames = load_agisoft_data()

    labeler = MeshLabeler(frames=frames, mesh=mesh, tree=tree,img_dir=image_folder, n_workers=20)
    #mesh = labeler.color_faces_from_images_all(image_raw_folder, '.jpg')
    mesh = labeler.color_faces_from_images_all(image_folder, '_pred.png')
    #labels, mesh = labeler.from_frame_interval(0, 120)
    #labels, mesh = labeler.from_all_frames()
    #labeler.write_labels(labels, 'test.txt')

    mesh.show()

    print('done')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
