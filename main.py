import numpy as np
import cv2
import trimesh
import xml.etree.ElementTree as ET
from Camera import Frame, Camera
from MeshLabeler import MeshLabeler
from aabbtree_mod import AABB, AABBTree
import json

mesh_file = './data/Sapelo_202106_run15/mesh.ply'
camera_file = './data/Sapelo_202106_run15/agisoft_cameras_Imaging.xml'
image_folder = './data/Sapelo_202106_run15/imaging_preds_3by2/'

# n_classes =9
# class_map = {  # RGB to Class
#     (0, 0, 0): -1,  # out of bounds
#     (255, 255, 255): 0,  # background
#     (150, 255, 14): 0,  # Background_alt
#     (127, 255, 140): 1,  # Spartina
#     (113, 255, 221): 2,  # dead Spartina
#     (99, 187, 255): 3,  # Sarcocornia
#     (101, 85, 255): 4,  # Batis
#     (212, 70, 255): 5,  # Juncus
#     (255, 56, 169): 6,  # Borrichia
#     (255, 63, 42): 7,  # Limonium
#     (255, 202, 28): 8  # Other
# }

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

# def maskrgb_to_class(mask):
#     h, w, channels = mask.shape[0], mask.shape[1], mask.shape[2]
#     mask_out = -1*np.ones((h, w), dtype=int)
#
#     for k in class_map:
#         matches = np.zeros((h, w ,channels), dtype=bool)
#
#         for c in range(channels):
#            matches[:,:,c] = mask[:,:,c] == k[c]
#
#         matches_total = np.sum(matches, axis=2)
#         valid_idx = matches_total == channels
#         mask_out[valid_idx] = class_map[k]
#
#     return mask_out
#
# def fractional_cover_from_selection(class_data):
#     pixel_count = []
#     for i in range(n_classes):
#         t = np.sum(class_data == i)
#         pixel_count.append(t)
#
#     return pixel_count/np.sum(pixel_count)



if __name__ == '__main__':

    mesh = load_mesh()
    vertices = mesh.vertices.view(np.ndarray)
    faces = mesh.faces.view(np.ndarray)

    tree = AABBTree()
    tree = tree.from_mesh_faces(mesh)

    cameras, frames = load_agisoft_data()

    labeler = MeshLabeler(frames=frames, mesh=mesh, tree=tree,img_dir=image_folder, n_workers=20)
 #   labels, mesh = labeler.from_frame_interval(0, 60)
    labels, mesh = labeler.from_all_frames()
    labeler.write_labels(labels, 'test.txt')

    mesh.show()

    print('done')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
