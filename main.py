import trimesh
import xml.etree.ElementTree as ET
from Camera import Frame, Camera
from MeshPlacer import MeshPlacer
from aabbtree import AABBTree
import time

""" script to place objects detected in images on mesh  """

mesh_file = './data/Sapelo_202106_run10/mesh.ply'
camera_file = './data/Sapelo_202106_run10/agisoft_cameras_Imaging.xml'

image_dir = './data/Sapelo_202106_run10/imaging/'

object_info = {
    'dir': './data/Sapelo_202106_run10/snail_preds/',
    'ext': '_preds.txt'

}

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
            _camera = Camera()
            _camera.load_agisoft(camera, version)
            cameras[_camera.id] = _camera

        frames_this_chunk = chunk.find('cameras') #my terminolgy 'frame' = agisoft 'camera'
        for frame in frames_this_chunk:
            _frame = Frame()
            _frame.load_agisoft(frame, cameras)
            frames.append(_frame)

    return cameras, frames


if __name__ == '__main__':

    mesh = load_mesh()
    tree = AABBTree()
    tree = tree.from_mesh_faces(mesh)

    cameras, frames = load_agisoft_data()

    object_placer = MeshPlacer(frames=frames, mesh=mesh, tree=tree, n_workers=8, obj_info=object_info, img_dir=image_dir)

    start = time.perf_counter()
    object_placer.place_objects_from_frames(start=0, stop=20)
    stop = time.perf_counter()
    dur = stop-start
    print('processing took : {} seconds'.format(dur))
