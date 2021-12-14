# assess fraction of mesh actually viewed in registered frames compared with what could possibly be visible
# by (approx) continuous registration of images. does this by following trajectory implied by registered frames and so
# there can't be any major uncompleted sections of the reconstruction (which is the case for current reconstructions)

from pycamgeom.camera import Camera
from pycamgeom.frame import Frame
from pycamgeom.aabbtree import AABBTree

import trimesh
import xml.etree.ElementTree as ET

mesh_file = './data/mesh.ply'
camera_file = './data/agisoft_cameras_Imaging.xml'



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

def faces_visible_in_frames(frames, mesh):



cameras, frames = load_agisoft_data(camera_file)
mesh = load_mesh(mesh_file)
#vertices = mesh.vertices.view(np.ndarray)
#faces = mesh.faces.view(np.ndarray)

tree = AABBTree()
tree = tree.from_mesh_faces(mesh)

# union of mesh faces visible in registered frames
visible_face_ids = faces_visible_in_frames(frames, mesh)
