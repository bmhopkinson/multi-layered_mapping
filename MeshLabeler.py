import numpy as np
import cv2
import copy
import re
import trimesh
import utils.helpers as h
import utils.image_processing as ip

parse_cover_key = re.compile('(.*)_(.*)')

DESCEND = 4  #number of levels to descend into the AABBtree, used to overcome issues with camera poses at global scale

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

class MeshLabeler():
    """ labels a mesh using registered image frames. search for mesh components visible in frame is  accelerated using aabb tree.
    class labels are typically from semantically segmented images which are decoded with a class map
    """
    def __init__(self, frames=None, mesh=None, tree=None, img_dir=[], n_workers=1):
        self.frames = frames
        self.mesh = copy.deepcopy(mesh)
        self.ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)
        self.vertices = []
        self.faces = []
        self.tree = tree  # aabb tree 
        self.img_dir = img_dir
        self.n_workers = n_workers
        self.manager = None
        self.run_function = h.run_concurrent

        if self.mesh is not None:
            self.vertices = mesh.vertices.view(np.ndarray)
            self.faces = mesh.faces.view(np.ndarray)

    def from_frame_interval(self, start=0, stop=0):
        """ determines fractional on faces visible in frames specified in interval. accelerated with multiprocessing"""
        frames_selection = self.frames[start:stop]
        cover_raw = self.run_function(self, self.process_frames_prediction, frames_selection, args=[], n_workers=self.n_workers)
        cover = h.collate_results(cover_raw, parse_cover_key)

        cover_avg = {}  # face_id: averaged_fractional_cover (over all observations of face)
        for face in cover:
            cover_avg[face] = np.mean(cover[face], axis=0)

        for face in cover_avg:   # color mesh based on one class for visualization purposes
            color = int(255 * cover_avg[face][1])
            self.mesh.visual.face_colors[face] = np.array([255 - color, 255, 255 - color, 255], dtype=np.uint8)

        return cover_avg, self.mesh

    def from_all_frames(self):
        """ determines fractional on faces visible in all registered frames"""
        start = 0
        stop = len(self.frames)
        return self.from_frame_interval(start, stop)

    def process_frames_prediction(self, frames, cover, args=[]):
        """ takes in a set of frames associated with a mesh and returns the fractional cover for each mesh face
        visible in the frames. tried to break this down further but holding onto image sections for even a bit results in enormous memory usage
        this function is used as multiprocessing target"""
        for frame in frames:
            print('working on {}, id: {}'.format(frame.label, frame.frame_id))
            hits, aabbs = frame.project_from_tree(self.tree, descend=DESCEND)
            hits_refined = self.refine_hits(frame, hits)

            img_pred_path = self.img_dir + frame.label + "_pred.png"
            img_pred = cv2.imread(img_pred_path)
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

            for face in hits_refined:
                triangle = hits_refined[face]
                selection, tri_coords = ip.extract_triangle_from_image(triangle, img_pred)
                class_data = ip.maskrgb_to_class(selection, class_map)
                fc = self.fractional_cover_from_selection(class_data)
                if(np.sum(np.isnan(fc)) != 0):  #this can legitimately occur on very rare occassions when the triangle is so thin that no pixels are selected
                    continue

                key = str(face) + "_" + str(frame.frame_id)
                cover[key] = np.array(fc, ndmin=2)

        return cover

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
              #  print('los not clear for {}'.format(face_id))

        return face_ids

    def fractional_cover_from_selection(self, class_data):
        """ returns vector indicating the fraction of image input (class_data) in each class (fractional cover in ecological terms)"""
        pixel_count = []
        for i in range(n_classes):
            t = np.sum(class_data == i)
            pixel_count.append(t)

        return pixel_count / np.sum(pixel_count)

    def write_labels(self, labels=None, file_name=None):
        """ writes out fractional cover by face data to text file"""
        fout = open(file_name, 'w')
        for face in labels:
            fout.write('{:d}'.format(face))
            for elm in labels[face]:
                fout.write('\t{:f}'.format(elm))
            fout.write('\n')

        fout.close()

    def color_faces_from_images_interval(self, start, stop, image_folder, ext, color_modifier=None, mode='Avg'):
        """ frontend for _color_faces_from_images() - colors mesh faces from frames in (start, stop) interval.
        breaks those frames into chunks and doles out to multiprocessing targets """
        frames_selection = self.frames[start:stop]

        args = [image_folder, ext, color_modifier]
        face_colors_raw = self.run_function(self, self._color_faces_from_images, frames_selection, args=args,
                                      n_workers=self.n_workers)
        face_colors_allviews = h.collate_results(face_colors_raw, parse_cover_key)

        face_colors = {}  # face_id: averaged_fractional_cover (over all observations of face)
        for face in face_colors_allviews:
            if mode == 'Avg':
                face_colors[face] = np.mean(face_colors_allviews[face], axis=0)
            elif mode == 'Single':
                face_colors[face] = face_colors_allviews[face][0, :]

        for face in face_colors:
            colors = face_colors[face]
            self.mesh.visual.face_colors[face] = np.array([colors[0], colors[1], colors[2], 255], dtype=np.uint8)

        fout = open('face_colors.txt', 'w')
        for face in face_colors:
            fout.write('{:d}'.format(face))
            for c in face_colors[face]:
                fout.write('\t{:d}'.format(int(c)))
            fout.write('\n')

        fout.close()

        return self.mesh

    def color_faces_from_images_all(self, image_folder, ext, color_mod=None):
        """ colors all mesh faces visible in frames based on images in image_folder """
        start = 0
        stop = len(self.frames)
        return self.color_faces_from_images_interval(start, stop, image_folder, ext, color_modifier=color_mod)

    def _color_faces_from_images(self, frames, face_colors, args=[]):
        """ colors (rgb) each visible face of the mesh with the average color in associated images whose poses
        are provided in frame and actual image are in image_folder - can either be rgb images or semantic segmentation.
        primarily for visualization. target for multiprocessing """
        image_folder = args[0]
        ext = args[1]
        color_mod = args[2]

        for frame in frames:
            print('working on {}, id: {}'.format(frame.label, frame.frame_id))
            hits, aabbs = frame.project_from_tree(self.tree, descend=DESCEND)
            hits_refined = self.refine_hits(frame, hits)

            img_path = image_folder + frame.label + ext
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            for face in hits_refined:
                triangle = hits_refined[face]
                selection, tri_coords = ip.extract_triangle_from_image(triangle, img)

                if color_mod is not None:
                    selection = color_mod(selection)

                mask = np.zeros((selection.shape[0], selection.shape[1]), dtype=np.uint8)
                cv2.fillConvexPoly(mask, tri_coords, 255)
                color = cv2.mean(selection, mask)
                key = str(face) + "_" + str(frame.frame_id)
                face_colors[key] = color[0:3]

        return face_colors





