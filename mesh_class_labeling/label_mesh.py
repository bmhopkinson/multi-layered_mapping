import os.path

import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from pycamgeom.camera import Camera
from pycamgeom.frame import Frame
from pycamgeom.aabbtree import AABBTree
import pycamgeom
print(pycamgeom.__file__)
from MeshLabeler import MeshLabeler

import time

""" script to label marsh mesh from images - uses semantically segmented images for class (plant) labeling and raw
     images for coloring mesh for visualizations """

MODE = 'Label_Interval'  #options 'Label_Interval', 'Label_All', 'Color_True', 'Color_Class'

mesh_file = './data/Sapelo_202106_run13/mesh_fine.ply'
camera_file = './data/Sapelo_202106_run13/agisoft_cameras_Imaging.xml'
image_classcolor_folder = './data/Sapelo_202106_run13/imaging_preds_20210909_model/'
image_truecolor_folder ='./data/Sapelo_202106_run13/imaging/'

# mesh_file = './data/mesh.ply'
# camera_file = './data/agisoft_cameras_Imaging.xml'
# image_classcolor_folder = './data/imaging_preds/'
# image_truecolor_folder ='./data/imaging/'

image_folder = image_classcolor_folder

outdir = './output'
if not os.path.isdir(outdir):
    os.mkdir(outdir)


def load_mesh(mesh_filename):
    mesh = trimesh.load_mesh(mesh_filename)
    return mesh

def load_agisoft_data(camera_filename):
    tree = ET.parse(camera_filename)
    root = tree.getroot()
    version = root.attrib['version']
    print(version)

    chunks = root.findall('chunk')
    cameras = {}
    frames = []

    for chunk in chunks:
        cameras_this_chunk = chunk.find('sensors')  #my terminolgy 'camera' = agisoft 'sensor'
        for camera in cameras_this_chunk:
            cam = Camera.load_agisoft(camera, version)
            cameras[cam.id] = cam

        frames_this_chunk = chunk.find('cameras') #my terminolgy 'frame' = agisoft 'camera'
        for frame in frames_this_chunk:
            _frame = Frame.load_agisoft(frame, cameras)
            frames.append(_frame)


    return cameras, frames

def remove_color_other(patch):
    #converts pixels labels as "other" (orange) to white (background) for visualization
    other_color = (255, 202, 28)
    white = (255, 255, 255)
    h, w, channels = patch.shape[0], patch.shape[1], patch.shape[2]
    matches = np.zeros((h, w, channels), dtype=bool)

    for c in range(channels):
        matches[:, :, c] = patch[:, :, c] == other_color[c]

    matches_total = np.sum(matches, axis=2)
    valid_idx = matches_total == channels
    patch[valid_idx] = white

    return patch


if __name__ == '__main__':

    cameras, frames = load_agisoft_data(camera_file)

    mesh = load_mesh(mesh_file)
    vertices = mesh.vertices.view(np.ndarray)
    faces = mesh.faces.view(np.ndarray)

    tree = AABBTree()
    tree = tree.from_mesh_faces(mesh)

    labeler = MeshLabeler(frames=frames, mesh=mesh, tree=tree, img_dir=image_folder, n_workers=20)

    start = time.perf_counter()
    if MODE == 'Label_All':
        labels, mesh = labeler.from_all_frames()
        labeler.write_labels(labels, os.path.join(outdir, 'fractional_cover_by_face.txt'))
    elif MODE == 'Label_Interval':
        labels, mesh = labeler.from_frame_interval(0, 40)
        labeler.write_labels(labels, os.path.join(outdir, 'fractional_cover_by_face.txt'))
    elif MODE == 'Color_Class':
        mesh = labeler.color_faces_from_images_all(image_folder, '_pred.png', remove_color_other)
    elif MODE == 'Color_True':
        mesh = labeler.color_faces_from_images_all(image_folder, '.jpg')
        #mesh = labeler.color_faces_from_images_interval(0, 20, image_folder, '.jpg')
    else:
        print("Error MODE not recognized.")

    stop = time.perf_counter()
    dur = stop-start
    print('processing took : {} seconds'.format(dur))
    mesh.show()

    print('done')
