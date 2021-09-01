import numpy as np
import trimesh
import xml.etree.ElementTree as ET
from Camera import Frame, Camera
from aabbtree_mod import AABB, AABBTree
import json

mesh_file = './data/mesh_data/Sapelo_202106_run13/mesh.ply'
camera_file = './data/mesh_data/Sapelo_202106_run13/agisoft_cameras_Imaging.xml'

def load_mesh():
    mesh = trimesh.load_mesh(mesh_file)
    return mesh

    #TODO: load Agisoft cameras, figure out AABB, start projecting


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

    cameras, frames = load_agisoft_data()

    frame_test = frames[100]
    camera_test = cameras[frame_test.camera_id]
    # bounds = [[0, 0.1],
    #            [2, 2.2],
    #             [7.5, 7.6]]
    #
    # frame_test.aabb_is_visible(bounds)

    tree = AABBTree()
    tree = tree.from_mesh_faces(mesh)

    hits, aabbs = frame_test.project_from_tree(tree, descend=4)
    vertex_ids = []
    for hit in hits:
        vertex_ids.extend(faces[hit,:].tolist())
    vertex_ids = set(vertex_ids)  #retain unique vertex ids

    vertex_hits_refined =[]
    for id in vertex_ids:
        vertex = vertices[id,:]
        vertex = np.append(vertex, 1.000)  # make homogeneous
        vertex = vertex.reshape((4, 1))

        valid, x_cam = frame_test.project(vertex)
        if valid:
            vertex_hits_refined.append({
                'vertex': vertex,
                'x': x_cam
            })

    mesh.visual.face_colors[hits] = np.array([255, 0, 0, 125], dtype = np.uint8)
    mesh.show()
    #
    vertex_hits_alt = []
    for vertex in vertices:
        vertex = np.append(vertex, 1.000)  # make homogeneous
        vertex = vertex.reshape((4, 1))

        valid, x_cam = frame_test.project(vertex)
        if valid:
            vertex_hits_alt.append({
                'vertex': vertex,
                'x': x_cam
            })

    #
    # mesh_wbbs = mesh
    # for bbox in aabbs:
    #     center = bbox.center()
    #     extents = bbox.distances
    #     box = trimesh.primitives.Box(
    #                      center=[0, 0, 0],
    #                      extents=[1, 1, 1])
    #     mesh_wbbs = mesh_wbbs + box
    #
    # mesh_wbbs.show()
    #
    #
    # combo = mesh + box
    # combo.show()
    print('done')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
