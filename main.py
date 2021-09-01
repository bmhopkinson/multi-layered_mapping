import numpy as np
import cv2
import trimesh
import xml.etree.ElementTree as ET
from Camera import Frame, Camera
from aabbtree_mod import AABB, AABBTree
import json

mesh_file = './data/Sapelo_202106_run13/mesh.ply'
camera_file = './data/Sapelo_202106_run13/agisoft_cameras_Imaging.xml'
image_folder = './data/Sapelo_202106_run13/imaging_preds_3by2/'

n_classes =9
class_map = {  # RGB to Class
    (0, 0, 0): -1,  # out of bounds
    (255, 255, 255): 0,  # background
    (150, 255, 14): 0,  # Background_alt
    (127, 255, 140): 1,  # Spartina
    (113, 255, 221): 2,  # dead Spartina
    (99, 187, 255): 3,  # Sarcocornia
    (101, 85, 255): 4,  # Batis
    (212, 70, 255): 5,  # Juncus
    (255, 56, 169): 6,  # Borrichia
    (255, 63, 42): 7,  # Limonium
    (255, 202, 28): 8  # Other
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
            cam = Camera()
            cam.load_agisoft(camera, version)
            cameras[cam.id] = cam

        frames_this_chunk = chunk.find('cameras') #my terminolgy 'frame' = agisoft 'camera'
        for frame in frames_this_chunk:
            _frame = Frame()
            _frame.load_agisoft(frame, cameras)
            frames.append(_frame)


    return cameras, frames

def maskrgb_to_class(mask):
    h, w, channels = mask.shape[0], mask.shape[1], mask.shape[2]
    mask_out = -1*np.ones((h, w), dtype=int)

    for k in class_map:
        matches = np.zeros((h, w ,channels), dtype=bool)

        for c in range(channels):
           matches[:,:,c] = mask[:,:,c] == k[c]

        matches_total = np.sum(matches, axis=2)
        valid_idx = matches_total == channels
        mask_out[valid_idx] = class_map[k]

    return mask_out

def fractional_cover_from_selection(class_data):
    pixel_count = []
    for i in range(n_classes):
        t = np.sum(class_data == i)
        pixel_count.append(t)

    return pixel_count/np.sum(pixel_count)



if __name__ == '__main__':

    mesh = load_mesh()
    vertices = mesh.vertices.view(np.ndarray)
    faces = mesh.faces.view(np.ndarray)

    tree = AABBTree()
    tree = tree.from_mesh_faces(mesh)

    cameras, frames = load_agisoft_data()

    frames_selection = frames[70:90]
    for frame in frames_selection:
        print('working on {}, id: {}'.format(frame.label, frame.frame_id))
        camera_test = cameras[frame.camera_id]
        hits, aabbs = frame.project_from_tree(tree, descend=4)
        hits_refined = {}
        for hit in hits:
            vertex_ids = faces[hit,:]
            face_vertices = vertices[vertex_ids, :]
            valid, pos = frame.project_triface(face_vertices)
            if valid:
                hits_refined[hit] = pos

        hits_idx = list(hits_refined.keys())
        mesh.visual.face_colors[hits_idx] = np.array([255, 0, 0, 125], dtype=np.uint8)

        img_pred_path = image_folder + frame.label + "_pred.png"
        img_pred = cv2.imread(img_pred_path)
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

        cover = {}
        for face in hits_refined:

            #select only the portion of image viewed in this face
            tri = hits_refined[face]
            tri.append(tri[0])
            tri = np.array(tri, dtype=int)
          #  tri = np.array([tri, tri[0]]) #close triangle
            mask = np.zeros((img_pred.shape[0], img_pred.shape[1]))
            cv2.fillConvexPoly(mask, tri, 1)
            mask = mask.astype(np.bool)
            selection = np.zeros_like(img_pred)
            selection[mask] = img_pred[mask]

            #crop for faster processing
            xy_min = np.amin(tri,axis=0)
            xy_max = np.amax(tri,axis=0)
            selection_crop = selection[xy_min[1]:xy_max[1], xy_min[0]:xy_max[0]]

        #    cv2.imshow('masked_img', cv2.cvtColor(selection_crop, cv2.COLOR_RGB2BGR))
        #    cv2.waitKey(0)

            #decode labels to calculate percent cover
            class_data = maskrgb_to_class(selection_crop)
            fc = fractional_cover_from_selection(class_data)
            cover[face] = fc
            #print('ehllo')

        for face in cover:
            mesh.visual.face_colors[face] = np.array([0, int(255*cover[face][4]), 0, 255], dtype = np.uint8)

    mesh.show()

    # frame_test2 = frames[103]
    # hits2, aabbs2 = frame_test2.project_from_tree(tree, descend=4)
    # hits_refined2 = {}
    # for hit in hits2:
    #     vertex_ids = faces[hit,:]
    #     face_vertices = vertices[vertex_ids, :]
    #     valid, pos = frame_test2.project_triface(face_vertices)
    #     if valid:
    #         hits_refined2[hit] = pos
    #
    # hits_idx2 = list(hits_refined2.keys())
    #
    # mesh.visual.face_colors[hits_idx2] = np.array([0, 255, 0, 125], dtype = np.uint8)
    # mesh.show()

    print('done')



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
