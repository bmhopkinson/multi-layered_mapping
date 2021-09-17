import multiprocessing as mp
import numpy as np
import trimesh
import sys
import copy
import math
import re
import json
import cv2
import os


DESCEND = 4  #number of levels to descend into the AABBtree, used to overcome issues with camera poses at global scale

parse_cover_key = re.compile('(.*)_(.*)')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class MeshPlacer():
    def __init__(self, frames=None, mesh=None, tree=None, obj_dir=[], img_dir=[], n_workers=1):
        self.frames = frames
        self.frame_from_id_dict = self.generate_frame_from_id_dict(frames)
        self.mesh = copy.deepcopy(mesh)
        self.ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)
        self.vertices = []
        self.faces = []
        self.tree = tree  # aabb tree
        self.obj_dir = obj_dir
        self.img_dir = img_dir
        self.n_workers = n_workers
        self.manager = None

        if self.mesh is not None:
            self.vertices = mesh.vertices.view(np.ndarray)
            self.faces = mesh.faces.view(np.ndarray)


    def allocate_faces_to_frames(self, start=0, stop=None):
        if stop is None:  #this means process all frames
            stop = len(self.frames)

        frames_selection = self.frames[start:stop]

        if self.manager is None:
            self.manager = mp.Manager()
        face_views = self.manager.dict()    # this will hold info about which faces are visible in which frames:  key 'faceid_frame_id', value: dist from center of projected face to camera projection center,
                                            # this structure is simpler to deal with in multiprocesing context and will be postproccesed later
        jobs = []
        for chunk in chunks(frames_selection, math.ceil(len(frames_selection) / self.n_workers)):
            j = mp.Process(target=self.visible_face_distances,
                           args=(chunk, face_views))
            j.start()
            jobs.append(j)

        for j in jobs:
            j.join()

        face_views_collated = self.collate_results(face_views)
        self.visualize_face_correspondences(face_views_collated, n=10)

        face_assigned_frame = {}
        for face in face_views_collated:
            frame_id = []
            min_dist = sys.float_info.max

            for view in face_views_collated[face]:
                if view[1] < min_dist:
                    frame_id = view[0]
                    min_dist = view[1]

            face_assigned_frame[face] = frame_id

        with open('face_views_collated.json', 'w') as f:
            json.dump(face_views_collated, f)

        with open('face_assigned_frames.json', 'w') as f:
            json.dump(face_assigned_frame, f)

        return face_assigned_frame

    def visible_face_distances(self, frames, face_views):
        for frame in frames:
            hits, aabbs = frame.project_from_tree(self.tree, descend=DESCEND)
            hits_refined = self.refine_hits(frame, hits)

            cx = frame.camera.cx  #projection center x coord
            cy = frame.camera.cy  #projection center y coord
            cam_center = np.array([cx, cy])

            for face in hits_refined:
                face_center = np.mean(hits_refined[face], axis=0)
                dist = np.linalg.norm(face_center-cam_center) #euclidean dist
                key = str(face) + "_" + str(frame.frame_id)
                face_views[key] = [frame.frame_id, dist]

        return face_views

    def collate_results(self, raw_ds):
        """takes raw dictionaries with multiple observations per face (different frames) and collates them by face"""
        collated_ds = {}
        for obs in raw_ds:
            m = parse_cover_key.search(obs)
            face = int(m.group(1))
            if face in collated_ds:
                collated_ds[face].append(raw_ds[obs])
            else:
                collated_ds[face] = [raw_ds[obs]]

        return collated_ds

    def refine_hits(self, frame, hits):
        """ takes faces that are likely visible (hits) in a frame based on crude method and refines those hits
        returning only faces (and associated image positions) that are visible in the frame"""
        hits_refined = {}
        for hit in hits:
            vertex_ids = self.faces[hit, :]
            face_vertices = self.vertices[vertex_ids, :]
            valid, pos = frame.project_triface(face_vertices)
            if valid:
                hits_refined[hit] = pos

        if hits_refined: #assuming we have potential visible faces, check for line of sight
            hits_refined = self.line_of_sight_test(hits_refined, frame.Twc[0:3, 3])

        return hits_refined


    def line_of_sight_test(self, face_ids, cam_center, forward=True):
        """ tests for a clear line of sight (not obstructed by mesh) between face_id and cam_center (in world coordinates);
        test can either be conducted by projecting a ray from camera center toward face center (forward) or backward- found
        forward to be slightly faster (i thought the opposite might be the case) and it's more conceptually straightfoward"""

        ray_orgs = np.empty((0, 3))
        ray_dirs = np.empty((0, 3))
        face_ids_ordered = []

        for face_id in face_ids:
            face_ids_ordered.append(face_id)

            # determine ray direction from cam_center to center of face (or reserve)
            vertex_ids = self.faces[face_id, :]
            face_vertices = self.vertices[vertex_ids, :]
            face_center = np.mean(face_vertices, axis=0)

            if forward:
                ray_dir = face_center - cam_center.reshape((1,3))
            else:
                ray_dir = cam_center.reshape((1, 3)) - face_center

            ray_dir = ray_dir/np.linalg.norm(ray_dir)
            ray_dirs = np.append(ray_dirs, ray_dir, axis=0)

            #ray origin
            if forward:
                ray_orgs = np.append(ray_orgs, cam_center.reshape((1, 3)), axis=0)
            else:
                #ray origin will be a point along the ray direction slightly off the face center
                v_deltas = face_vertices - face_vertices[[2, 0, 1], :]
                edge_len_mean = np.mean(np.linalg.norm(v_deltas, axis=1))
                ray_org = face_center + 0.05*edge_len_mean*ray_dir
                ray_orgs = np.append(ray_orgs, ray_org, axis=0)

        #conduct intersection test
        intersection_results = self.ray_mesh_intersector.intersects_first(ray_orgs, ray_dirs)

        for face_id, intersect_id in zip(face_ids_ordered, intersection_results):
            remove = False
            if forward:
                if face_id != intersect_id:
                    remove = True
            else:
                if intersect_id != -1 and intersect_id != face_id:
                    remove = True

            if remove:
                face_ids.pop(face_id)

        return face_ids

    def visualize_face_correspondences(self, face_views_collated, n=10):
        """draws outline of projected face onto images it projects into. used to qualitatively assess consistency of image registration"""

        i = 0

        N_GAP = 2000
        j_gap = 0
        in_gap = False

        output_folder ='./output/'
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        for face in face_views_collated:
            if i > n:
                break

            views = face_views_collated[face]

            if len(views) > 2:

                if not in_gap:
                    i = i + 1
                    vertex_ids = self.faces[face, :]
                    face_vertices = self.vertices[vertex_ids, :]
                    for view in views:
                        #determine bounds of face in this frame
                        frame_id = view[0]
                        frame = self.frame_from_id(frame_id)
                        valid, pos = frame.project_triface(face_vertices)

                        #draw bounds of face on this fram
                        img_path = self.img_dir + frame.label + ".jpg"
                        img = cv2.imread(img_path)

                        closed = True
                        color = [255, 0, 0]
                        thickness = 20
                        pos = pos.astype(np.int32)
                        pos = pos.reshape(-1, 1, 2)

                        img = cv2.polylines(img, [pos], closed, color, thickness)
                        outpath = output_folder + str(face) + "_" + frame.label + '.jpg'
                        cv2.imwrite(outpath,img)

                        in_gap = True

                else:
                    j_gap = j_gap + 1

                    if j_gap < N_GAP:
                        continue
                    else:
                        in_gap = False
                        j_gap = 0

    def generate_frame_from_id_dict(self, frames):
        frame_from_id = {}
        for i, frame in enumerate(frames):
            frame_from_id[frame.frame_id] = i

        return frame_from_id

    def frame_from_id(self, frame_id):
        return self.frames[self.frame_from_id_dict[frame_id]]




