import multiprocessing as mp
import numpy as np
import trimesh
import pandas
import sys
import copy
import math
import re
import json
import cv2
import os

#https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle

DESCEND = 4  #number of levels to descend into the AABBtree, used to overcome issues with camera poses at global scale

parse_cover_key = re.compile('(.*)_(.*)')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]




class MeshPlacer():
    def __init__(self, frames=None, mesh=None, tree=None, obj_info={}, img_dir=[], n_workers=1):
        self.frames = frames
        self.frame_from_id_dict = self.generate_frame_from_id_dict(frames)
        self.mesh = copy.deepcopy(mesh)
        self.ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)
        self.vertices = []
        self.faces = []
        self.faces_assigned = {}
        self.tree = tree  # aabb tree
        self.objects = []
        self.objects_assigned = {}
        self.img_dir = img_dir
        self.n_workers = n_workers
        self.manager = None

        if self.mesh is not None:
            self.vertices = mesh.vertices.view(np.ndarray)
            self.faces = mesh.faces.view(np.ndarray)

        if obj_info:
            self.objects = self.load_objects(obj_info)


    def load_objects(self, obj_info):
        object_data = {}
        for frame in self.frames:
            obj_file = obj_info['dir'] + frame.label + obj_info['ext']
            frame_data = pandas.read_csv(obj_file, sep='\t')
            frame_data['x_c'] = (frame_data['x_min'] + frame_data['x_max']) / 2
            frame_data['y_c'] = (frame_data['y_min'] + frame_data['y_max']) / 2
            object_data[frame.frame_id] = frame_data

       # print('done')
        return object_data

    def allocate_faces_to_frames(self, start=0, stop=None):
        if stop is None:  #this means process all frames
            stop = len(self.frames)

        frames_selection = self.frames[start:stop]
        results = self.run_concurrent(self.visible_face_distances, frames_selection, args=[])
        face_views_collated = self.collate_results(results)
       # self.visualize_face_correspondences(face_views_collated, n=10)

        faces_assigned_frame = {}
        for face in face_views_collated:
            frame_id = []
            min_dist = sys.float_info.max

            for view in face_views_collated[face]:
                if view[1] < min_dist:
                    frame_id = view[0]
                    min_dist = view[1]


            if frame_id in faces_assigned_frame:
                faces_assigned_frame[frame_id].append(face)
            else:
                faces_assigned_frame[frame_id] = [face]

        self.visualize_face_assignments(faces_assigned_frame, n=40)

        # with open('face_views_collated.json', 'w') as f:
        #     json.dump(face_views_collated, f)
        #
        # with open('face_assigned_frames.json', 'w') as f:
        #     json.dump(faces_assigned_frame, f)

        self.faces_assigned = faces_assigned_frame
        return faces_assigned_frame

    def visible_face_distances(self, frames, results, args):
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
                results[key] = [frame.frame_id, dist]

        return results

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

    def find_objects_in_faces(self):
        frame_ids = list(self.faces_assigned.keys())
        all_objs = self.run_concurrent(self._find_objects_in_faces, frame_ids)

        # for frame_id in self.faces_assigned:
        #     all_objs[frame_id] = {}
        #     frame = self.frame_from_id(frame_id)
        #     for face in self.faces_assigned[frame_id]:
        #         objs = self.objects_in_face(frame, face)
        #         if objs:
        #             all_objs[frame_id][face] = objs

        self.objects_assigned = all_objs

       # with open('assigned_objects.json', 'w') as f:
       #     json.dump(all_objs, f)

        self.visualize_object_assignments(n=40)
        return all_objs

    def _find_objects_in_faces(self, frame_ids, results, args):
        for frame_id in frame_ids:
            #results[frame_id] = {}
            results_frame = {}
            frame = self.frame_from_id(frame_id)
            for face in self.faces_assigned[frame_id]:
                objs = self.objects_in_face(frame, face)
                if objs:
                    results_frame[face] = objs

            results[frame_id] = results_frame

        return results

    def objects_in_face(self, frame, face_id):
        objs_valid = []
        objs_frame = self.objects[frame.frame_id]
        vertex_ids = self.faces[face_id, :]
        face_vertices = self.vertices[vertex_ids, :]
        valid, pos = frame.project_triface(face_vertices)

        for i, row in objs_frame.iterrows():
            obj_center = np.array([row['x_c'], row['y_c']])
            if self.is_in_triangle(obj_center, pos):
              #  print('obj in triangle, obj center: {}'.format(obj_center))
              #  print('triangle bounds: {}'.format(pos))
                objs_valid.append(obj_center)

        return objs_valid

    def is_in_triangle(self, pt, tri):
        '''use half plane method to determine if pt is in the triangle. conceptually traverse edges of triangle and test if point is to
        right or left of edge using cross product. a point in the triangle is either always on left or always on right as edges are traversed so
        cross products must all be positive or all be negative. see https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle'''
        p1 = self.cross_product(pt, tri[0, :], tri[1, :])
        p2 = self.cross_product(pt, tri[1, :], tri[2, :])
        p3 = self.cross_product(pt, tri[2, :], tri[0, :])

        if p1 < 0 and p2 < 0 and p3 < 0:
            return True
        elif p1 > 0 and p2 > 0 and p3 > 0:
            return True
        else:
            return False

    def cross_product(self, pt, tri_1, tri_2):
        res = (pt[0] - tri_2[0])*(tri_1[1] - tri_2[1] ) - (tri_1[0] - tri_2[0])*(pt[1] - tri_2[1])
        return res

    def generate_frame_from_id_dict(self, frames):
        frame_from_id = {}
        for i, frame in enumerate(frames):
            frame_from_id[frame.frame_id] = i

        return frame_from_id

    def frame_from_id(self, frame_id):
        return self.frames[self.frame_from_id_dict[frame_id]]

    def run_concurrent(self, func=[], data_in=[], args=[], ):
        if self.manager is None:
            self.manager = mp.Manager()
        results = self.manager.dict()  # this will hold info about which faces are visible in which frames:  key 'faceid_frame_id', value: dist from center of projected face to camera projection center,
        # this structure is simpler to deal with in multiprocesing context and will be postproccesed later
        jobs = []
        for chunk in chunks(data_in, math.ceil(len(data_in) / self.n_workers)):
            j = mp.Process(target=func,
                           args=(chunk, results, args))
            j.start()
            jobs.append(j)

        for j in jobs:
            j.join()

        return results.copy()  #convert to normal dictionary

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

    def visualize_face_assignments(self, face_assignments, n=10):
        """ for selected frames (n total), draws outline of all faces assinged to this frame after projection into image. used to qualitatively assess assingment process"""

        #prepare output directory
        output_folder = './output/'
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        i = 0
        for frame_id in face_assignments:
            if i > n:
                break

            frame = self.frame_from_id(frame_id)  #get relevant frame and load image
            img_path = self.img_dir + frame.label + ".jpg"
            img = cv2.imread(img_path)

            #draw projected assigned faces on img
            for face in face_assignments[frame_id]:
                vertex_ids = self.faces[face, :]
                face_vertices = self.vertices[vertex_ids, :]
                valid, pos = frame.project_triface(face_vertices)

                #draw face on img
                closed = True
                color = [255, 0, 0]
                thickness = 5
                pos = pos.astype(np.int32)
                pos = pos.reshape(-1, 1, 2)
                img = cv2.polylines(img, [pos], closed, color, thickness)

            outpath = output_folder + frame.label + '_assigned_faces.jpg'
            cv2.imwrite(outpath, img)
            i = i + 1


    def visualize_object_assignments(self, n=10):
        # prepare output directory
        output_folder = './output/'
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        i = 0
        for frame_id in self.objects_assigned:
            if i > n:
                break

            frame = self.frame_from_id(frame_id)  # get relevant frame and load image
            img_path = self.img_dir + frame.label + ".jpg"
            img = cv2.imread(img_path)

            # draw projected assigned faces on img
            for face in self.objects_assigned[frame_id]:
                vertex_ids = self.faces[face, :]
                face_vertices = self.vertices[vertex_ids, :]
                valid, pos = frame.project_triface(face_vertices)

                # draw face on img
                closed = True
                color = [255, 0, 0]
                thickness = 5
                pos = pos.astype(np.int32)
                pos = pos.reshape(-1, 1, 2)
                img = cv2.polylines(img, [pos], closed, color, thickness)

                for obj in self.objects_assigned[frame_id][face]:
                   # print('obj: {}'.format(type(obj)))
                   # print('obj[0]: {}, obj[1]: {}'.format(obj[0], obj[1]))
                    img = cv2.circle(img, obj.astype(np.int32), 5, color, thickness=3)

            outpath = output_folder + frame.label + '_assigned_objects.jpg'
            cv2.imwrite(outpath, img)
            i = i + 1


