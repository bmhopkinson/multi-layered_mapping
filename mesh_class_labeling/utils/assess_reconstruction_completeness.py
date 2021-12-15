# assess fraction of mesh actually viewed in registered frames compared with what could possibly be visible
# by (approx) continuous registration of images. does this by following trajectory implied by registered frames and so
# there can't be any major uncompleted sections of the reconstruction (which is the case for current reconstructions)

from pycamgeom.camera import Camera
from pycamgeom.frame import Frame
from pycamgeom.aabbtree import AABBTree
from pycamgeom.projector import Projector

from helpers import run_concurrent, run_singlethreaded

import copy
import trimesh
import numpy as np
import xml.etree.ElementTree as ET

mesh_file = '../data/UGA_1_1/mesh.ply'
camera_file = '../data/UGA_1_1/agisoft_cameras_Imaging.xml'


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


def find_visible_faces(frames, results, projector):
    for frame in frames:
        vis_in_frame = projector.find_visible_faces(frame)
        results.update(vis_in_frame) #note: this will overwrite duplicate keys, but for this application that's fine


if __name__ == "__main__":

    cameras, frames = load_agisoft_data(camera_file)
    mesh = load_mesh(mesh_file)
    vertices = mesh.vertices.view(np.ndarray)
    faces = mesh.faces.view(np.ndarray)

    tree = AABBTree()
    tree = tree.from_mesh_faces(mesh)
    projector = Projector(faces, vertices, mesh, tree, descend=4)
    print('finished loading frame and mesh data and constructing AABB tree')

    # union of mesh faces visible in registered frames
    visible_face_ids = run_concurrent(None, find_visible_faces, frames, args=projector, n_workers=20)
    print('found faces seen in registered frames')

    # find typical distance between frames and use to interpolate frame positions when there are gaps
    pos_prev = None
    interframe_dists = np.empty((0,), dtype=np.double)
    for frame in frames:
        pos = frame.Twc[0:3,3]
        if pos_prev is not None:
            delta = np.linalg.norm(pos-pos_prev)
            interframe_dists = np.append(interframe_dists, delta)
        pos_prev = pos

    dist_char = np.median(interframe_dists) #characteristic interframe distance

    frames_aug = copy.deepcopy(frames)
    pos_prev = None
    n_inserted = 0
    for i, frame in enumerate(frames):
        pos = frame.Twc[0:3, 3]
        if pos_prev is not None:
            delta = np.linalg.norm(pos - pos_prev)
            rel_dist = delta/dist_char
            if rel_dist > 1.5:
                n_interframes = int(rel_dist)
                delta_vec = (pos - pos_prev)/(n_interframes+1)
                base_frame = frames[i-1]
                Twc_base = np.copy(base_frame.Twc)
                pos_base = Twc_base[0:3,3]

                for j in range(1, n_interframes+1):
                    pos_inter = delta_vec*j + pos_base
                    Twc_inter = np.copy(Twc_base)
                    Twc_inter[0:3,3] = pos_inter
                    frame_inter = copy.deepcopy(base_frame)
                    frame_inter.set_pose(Twc_inter)
                    insertion_idx = i + n_inserted
                    frames_aug.insert(insertion_idx, frame_inter)
                    n_inserted = n_inserted + 1

        pos_prev = pos

    print('augmented frames to cover gaps in trajectory')

    max_visible_face_ids = run_concurrent(None, find_visible_faces, frames_aug, args=projector, n_workers=20)
    print('found faces seen in augmented frames')

    print('number of actually visible faces:{}'.format(len(visible_face_ids)))
    print('number of potentially visible faces:{}'.format(len(max_visible_face_ids)))
