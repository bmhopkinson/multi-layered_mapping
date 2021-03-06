import numpy as np
import pandas
import sys
import copy
import re
import cv2
import os
import utils.helpers as h

from pycamgeom.projector import Projector


DESCEND = 4  #number of levels to descend into the AABBtree, used to overcome issues with camera poses at global scale

parse_cover_key = re.compile('(.*)_(.*)')

class MeshPlacer():
    """ the MeshPlacer class places objects detected in registered 2D images onto a 3D mesh via backprojection.
        it avoids placing duplicate object detections by allocating mesh elements to individual frames. In each image,
        only objects whose center lies within the bounds of the allocated mesh elements are placed on the mesh.

        the primary public interface is place_objects_from_frames()
        MeshPlacer objects are constructed with a set of registered images ("frames"), a corresponding triangular mesh,
        an AABB (axis-aligned bounding box) tree to accelerate searches on the mesh, information about where the object
        data is to be found (obj_info), optionally an image directory (only used for visualization/troubleshooting), and
        number of workers to use in multiprocessing steps.
        certain time consuming steps can be accelerated with multiprocessing by setting self.run_function = h.run_concurrent.
        for debugging this can be swapped with h.run_singlethreaded
    """
    def __init__(self, frames, mesh=None, tree=None,  mode='face_allocation', obj_info=None, img_dir=None, n_workers=1):
        self.frames = frames
        self.frame_from_id_dict = self.generate_frame_from_id_dict(frames)
        self.mesh = copy.deepcopy(mesh)
        self.vertices = []
        self.faces = []
        self.tree = tree            # aabb tree
        self.objects_imgs = {}      # raw object data - all objects in each frame (key: frame_id, value: panda table of object data)
        self.objects_world = []     # objects backprojected into world coordinates. each item in list has world position 'x_world' and class type 'type'
        self.mode = mode            # approach to avoiding duplicate objects
        self.n_workers = n_workers
        self.manager = None   # data structure manager for multiprocessing operations
        self.run_function = h.run_concurrent

        if self.mesh is not None:
            self.vertices = mesh.vertices.view(np.ndarray)
            self.faces = mesh.faces.view(np.ndarray)

        self.projector = Projector(self.faces, self.vertices, mesh, tree, descend=4)

        if img_dir is None:
            self.img_dir = './'
        else:
            self.img_dir = img_dir

        if obj_info:
            self.objects_imgs = self.load_objects(obj_info)
            print('loaded object data')


    def load_objects(self, obj_info):
        """loads object data from file, obj_info specifies the directory and extension
           to find object detections per frame"""
        object_data = {}
        for frame in self.frames:
            obj_file = obj_info['dir'] + frame.label + obj_info['ext']
            frame_data = pandas.read_csv(obj_file, sep='\t')
            if 'x_c' not in frame_data.columns:
                frame_data['x_c'] = (frame_data['x_min'] + frame_data['x_max']) / 2
                frame_data['y_c'] = (frame_data['y_min'] + frame_data['y_max']) / 2
            object_data[frame.frame_id] = frame_data

        return object_data

    def place_objects_from_frames(self, start=0, stop=None, outfile='out.txt'):
        """" primary public interface - places objects on mesh from frames start to stop, or all if not specified"""
        if stop is None:  #this means process all frames
            stop = len(self.frames)

        if self.mode == 'face_allocation':
            faces_assigned = self.allocate_faces_to_frames(start, stop)
            print('allocated faces to frames')
            objects_to_place = self.find_objects_in_faces(faces_assigned)
            print('found objects in faces')
        elif self.mode == 'unique_id':
            objects_to_place = self.objects_for_uniqueid_placement()
        else:
            print('MODE NOT RECOGNIZED!')

        print('identified objects to place on mesh')

        self.backproject_objects_to_mesh(objects_to_place)

        if self.mode == 'unique_id':
            self.objects_world = self.merge_duplicate_objects()

        self.write_placed_objects(outfile)

    def allocate_faces_to_frames(self, start=0, stop=None):
        """ allocates mesh elements (faces) to individual frames to avoid duplicate object placement.
            assigns faces to frame in which their projected location is closest to center of image, which generally
            provides best (and least distorted) view.
            """
        if stop is None:  #this means process all frames
            stop = len(self.frames)

        frames_selection = self.frames[start:stop]

        # for each visible face determine distances from center in all frames it is viewed in, this is slow,
        # so can be accelerated by running concurrently
        results = self.run_function(self, self.visible_face_distances, frames_selection, n_workers=self.n_workers)
        face_views_collated = h.collate_results(results, parse_cover_key)

       # self.visualize_face_correspondences(face_views_collated, n=10)

        # determine frame in which face is closest to center (minimum distance) and assign face to that frame
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

        return faces_assigned_frame   # faces assigned to frames (key: frame_id, value: assigned faces)

    def visible_face_distances(self, frames, results, args):
        """ determines distances from image center for faces visible in frames """
        for frame in frames:
            # find faces visible in frame, results from project_from_tree (fast) are approximate so need to be 'refined'
            hits_refined = self.projector.find_visible_faces(frame)

            cx = frame.camera.cx  #projection center x coord
            cy = frame.camera.cy  #projection center y coord
            cam_center = np.array([cx, cy])

            for face in hits_refined:
                face_center = np.mean(hits_refined[face], axis=0)
                dist = np.linalg.norm(face_center-cam_center) #euclidean dist
                key = str(face) + "_" + str(frame.frame_id)
                results[key] = [frame.frame_id, dist]

        return results

    def find_objects_in_faces(self, faces_assigned):
        """ for each frame that has assigned faces, finds objects whose center is within assigned faces """
        frame_ids = list(faces_assigned.keys())
        objects_assigned = self.run_function(self, self._find_objects_in_faces,  frame_ids, args=[faces_assigned], n_workers=self.n_workers)
        self.visualize_object_assignments(objects_assigned, n=40)
        return objects_assigned

    def _find_objects_in_faces(self, frame_ids, results, args):
        """ working target for find_objects_in_faces
            steps through frames and calls 'objects_in_face()' on each face assigned to frame, collecting results"""

        faces_assigned = args[0]

        for frame_id in frame_ids:
            #results[frame_id] = {}
         #   results_frame = {}
            results_frame = []
            frame = self.frame_from_id(frame_id)
            for face in faces_assigned[frame_id]:
                objs = self.objects_in_face(frame, face)
                if objs:
                    results_frame.extend(objs)

            results[frame_id] = results_frame

        return results

    def objects_in_face(self, frame, face_id):
        """  for face_id assigned to frame, determines if any detected object centers are within the bounds of
             the projected face
        """

        objs_valid = []
        objs_frame = self.objects_imgs[frame.frame_id]
        vertex_ids = self.faces[face_id, :]
        face_vertices = self.vertices[vertex_ids, :]
        valid, pos = frame.project_triface(face_vertices)  # pos holds projected bounds of face_id in frame

        for i, row in objs_frame.iterrows():   # should convert triangluar bounds to square and throw out any objects not within this square (but runs fast enough right now)
            obj_center = np.array([row['x_c'], row['y_c']])
            if self.is_in_triangle(obj_center, pos):
                objs_valid.append({'type': row['type'].astype(int), 'img_xy': obj_center, 'unique_id': None, 'face_id': face_id})

        return objs_valid

    def is_in_triangle(self, pt, tri):
        """ use half plane method to determine if pt is in the triangle. conceptually traverse edges of triangle and test if point is to
        right or left of edge using cross product. a point in the triangle is either always on left or always on right as edges are traversed so
        cross products must all be positive or all be negative. see https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
        """
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
        res = (pt[0] - tri_2[0]) * (tri_1[1] - tri_2[1]) - (tri_1[0] - tri_2[0]) * (pt[1] - tri_2[1])
        return res

    def objects_for_uniqueid_placement(self):
        objects_to_place = {}
        for frame_id in self.objects_imgs:
            _objects = []
            for i, row in self.objects_imgs[frame_id].iterrows():
                obj_center = np.array([row['x_c'], row['y_c']])
                _objects.append({'type': row['type'].astype(int), 'img_xy': obj_center, 'unique_id': row['unique_id'].astype(int)})

            if _objects:
                objects_to_place[frame_id] = _objects

        return objects_to_place


    def backproject_objects_to_mesh(self, objects_to_place):
        """ takes each assigned object and backprojects it onto the mesh
            this could be parallelized but right now it doesn't take very long (~1000s of objects to backproject)
        """
        for frame_id in objects_to_place:
            frame = self.frame_from_id(frame_id)

            ray_to_obj = []
            imgpts = []
            for obj in objects_to_place[frame_id]:
                ray_to_obj.append(obj)
                imgpts.append(obj['img_xy'])

            locations, ray_idx, face_ids = self.projector.backproject_imgpts_to_mesh(frame, imgpts)

            for loc, ray_id in zip(locations, ray_idx):
                self.objects_world.append({'x_world': loc, 'type': ray_to_obj[ray_id]['type'], 'unique_id': ray_to_obj[ray_id]['unique_id']})


    def merge_duplicate_objects(self):
        #collate objects by unique_id
        objects_unique = {}
        objects_merged = []
        for obj in self.objects_world:
            _id = obj['unique_id']
            if _id in objects_unique:
                objects_unique[_id].append(obj)
            else:
                objects_unique[_id] = [obj]

        #merge by averaging 3D positions - object may no longer lie on mesh surface
        for obj_id in objects_unique:
            pos = np.empty((0, 3), dtype=np.float)
            for instance in objects_unique[obj_id]:
                pos = np.append(pos, np.expand_dims(instance['x_world'], axis=0), axis=0)  #may need np.expand_dims(instance['x_world'], axis=0)

            print('objid: {}, pos: {}'.format(obj_id, pos))
            pos_best = np.mean(pos, axis=0)
            obj_merged = objects_unique[obj_id][0]  #all data in duplicate objects should be identical except position
            obj_merged['x_world'] = pos_best
            objects_merged.append(obj_merged)

        return objects_merged

    def write_placed_objects(self, out_path):
        fout = open(out_path, 'w')
        for obj in self.objects_world:
            fout.write('{:d}'.format(obj['type']))

            if obj['unique_id'] is not None:
                fout.write('\t{:d}'.format(obj['unique_id']))

            for x in obj['x_world']:
                fout.write('\t{:f}'.format(x))
            fout.write('\n')

        fout.close()


    def generate_frame_from_id_dict(self, frames):
        frame_from_id = {}
        for i, frame in enumerate(frames):
            frame_from_id[frame.frame_id] = i

        return frame_from_id

    def frame_from_id(self, frame_id):
        return self.frames[self.frame_from_id_dict[frame_id]]

    ###### FUNCTIONS BELOW ARE FOR VISUALIZATION AND TROUBLESHOOTING #######
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


    def visualize_object_assignments(self, objects_assigned, n=10):
        # prepare output directory
        output_folder = './output/'
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        i = 0
        for frame_id in objects_assigned:
            if i > n:
                break

            frame = self.frame_from_id(frame_id)  # get relevant frame and load image
            img_path = self.img_dir + frame.label + ".jpg"
            img = cv2.imread(img_path)

            # draw projected assigned faces on img
            for obj in objects_assigned[frame_id]:
                vertex_ids = self.faces[obj['face_id'], :]
                face_vertices = self.vertices[vertex_ids, :]
                valid, pos = frame.project_triface(face_vertices)

                # draw face on img
                closed = True
                color = [255, 0, 0]
                thickness = 5
                pos = pos.astype(np.int32)
                pos = pos.reshape(-1, 1, 2)
                img = cv2.polylines(img, [pos], closed, color, thickness)
                img = cv2.circle(img, tuple(obj['img_xy'].astype(np.int32)), 5, tuple(color), thickness=3)

            outpath = output_folder + frame.label + '_assigned_objects.jpg'
            cv2.imwrite(outpath, img)
            i = i + 1


