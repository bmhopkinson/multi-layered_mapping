import trimesh
import os
import xml.etree.ElementTree as ET
from pycamgeom.camera import Camera
from pycamgeom.frame import Frame
from pycamgeom.aabbtree import AABBTree
from MeshPlacer import MeshPlacer

import time

""" script to place objects detected in images on mesh  """

mesh_file = './data/Sapelo_202106_run15/mesh_fine.ply'
camera_file = './data/Sapelo_202106_run15/agisoft_cameras_Imaging.xml'

image_dir = './data/Sapelo_202106_run15/imaging/'

object_info = {
    'dir': './data/Sapelo_202106_run15/snail_preds/',
    'ext': '_preds.txt'

}
mode = 'face_allocation'


# mesh_file = './data/Sapelo_202110_run2_hyslam/mesh_imaging.ply'
# camera_file = './data/Sapelo_202110_run2_hyslam/agisoft_cameras_Imaging.xml'
#
# image_dir = []#'./data/Sapelo_202110_run3/imaging/'
#
# object_info = {
#     'dir': './data/Sapelo_202110_run2_hyslam/AprilTags/',
#     'ext': '_tags.txt'
# }

#mode = 'unique_id'

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

    chunks = root.findall('chunk')
    cameras = {}
    frames = []

    for chunk in chunks:
        cameras_this_chunk = chunk.find('sensors')  #my terminolgy 'camera' = agisoft 'sensor'
        for camera in cameras_this_chunk:
            _camera = Camera.load_agisoft(camera, version)
            cameras[_camera.id] = _camera

        frames_this_chunk = chunk.find('cameras') #my terminolgy 'frame' = agisoft 'camera'
        for frame in frames_this_chunk:
            _frame = Frame.load_agisoft(frame, cameras)
            frames.append(_frame)

    return cameras, frames


if __name__ == '__main__':

    cameras, frames = load_agisoft_data(camera_file)

    mesh = load_mesh(mesh_file)
    if mode == 'face_allocation':
        tree = AABBTree()
        tree = tree.from_mesh_faces(mesh)
        print('finished constructing AABBtree for mesh')
    else:
        tree = []

    object_placer = MeshPlacer(frames=frames, mesh=mesh, tree=tree, n_workers=8, mode=mode,
                               obj_info=object_info, img_dir=image_dir)

    start = time.perf_counter()
    object_placer.place_objects_from_frames(start=0, stop=None, outfile=os.path.join(outdir, 'placed_objects.txt'))
    stop = time.perf_counter()
    dur = stop-start
    print('processing took : {} seconds'.format(dur))
