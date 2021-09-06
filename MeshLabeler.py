import numpy as np
import cv2
import copy
import math
import re
import multiprocessing as mp

parse_cover_key = re.compile('(.*)_(.*)')

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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class MeshLabeler():
	""" labels a mesh using registered image frames. search for mesh components visible in frame is  accelerated using aabb tree.
		class labels are typically from semantically segmented images which are decoded wit a class map"""
    def __init__(self, frames=None, mesh=None, tree=None, img_dir=[], n_workers=1):
        self.frames = frames
        self.mesh = copy.deepcopy(mesh)
        self.vertices = []
        self.faces = []
        self.tree = tree  # aabb tree 
        self.img_dir = img_dir
        self.n_workers = n_workers
        self.manager = None

        if self.mesh is not None:
            self.vertices = mesh.vertices.view(np.ndarray)
            self.faces = mesh.faces.view(np.ndarray)

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

        return hits_refined

    def extract_triangle_from_image(self, triangle, image):
        """select only the portion of  prediction image within this triangle and covert rgb codes to class data"""

        #crop image to fit triangle and recompute triangle coordinates
        triangle = np.array(triangle, dtype=int)
        xy_min = np.amin(triangle, axis=0).astype(int)
        xy_max = np.amax(triangle, axis=0).astype(int)
        image_crop = image[xy_min[1]:xy_max[1], xy_min[0]:xy_max[0]]
        tri_crop = triangle
        tri_crop[:, 0] = tri_crop[:, 0] - xy_min[0]     #revise x-coords
        tri_crop[:, 1] = tri_crop[:, 1] - xy_min[1]     #revise y-coords
        tri_crop = np.append(tri_crop, np.array(tri_crop[0,:], ndmin=2), axis=0) #close triangle
        mask = np.zeros((image_crop.shape[0], image_crop.shape[1]))

        #create mask over triangular region
        cv2.fillConvexPoly(mask, tri_crop, 1)
        mask = mask.astype(np.bool)
        selection = np.zeros_like(image_crop)
        selection[mask] = image_crop[mask]  #extract predictions within mask region
      #  cv2.imshow('selection', selection)
      #  cv2.waitKey(0)

        return selection, tri_crop

    def maskrgb_to_class(self, mask):
    """ decode rgb mask to classes using class map"""
        h, w, channels = mask.shape[0], mask.shape[1], mask.shape[2]
        mask_out = -1 * np.ones((h, w), dtype=int)

        for k in class_map:
            matches = np.zeros((h, w, channels), dtype=bool)

            for c in range(channels):
                matches[:, :, c] = mask[:, :, c] == k[c]

            matches_total = np.sum(matches, axis=2)
            valid_idx = matches_total == channels
            mask_out[valid_idx] = class_map[k]

        return mask_out

    def fractional_cover_from_selection(self, class_data):
    """ returns vector indicating the fraction of image input (class_data) in each class (fractional cover in ecological terms)"""
        pixel_count = []
        for i in range(n_classes):
            t = np.sum(class_data == i)
            pixel_count.append(t)

        return pixel_count / np.sum(pixel_count)

    def process_frames_prediction(self, frames, cover):
    """ takes in a set of frames associated with a mesh and returns the fractional cover for each mesh face
        visible in the frames. tried to break this down further but holding onto image sections for even a bit results in enormous memory usage  
        this function is used as multiprocessing target"""
        for frame in frames:
            print('working on {}, id: {}'.format(frame.label, frame.frame_id))
            hits, aabbs = frame.project_from_tree(self.tree, descend=4)
            hits_refined = self.refine_hits(frame,hits)

            img_pred_path = self.img_dir + frame.label + "_pred.png"
            img_pred = cv2.imread(img_pred_path)
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

            for face in hits_refined:
                triangle = hits_refined[face]
                selection, tri_coords = self.extract_triangle_from_image(triangle, img_pred)
                class_data = self.maskrgb_to_class(selection)
                fc = self.fractional_cover_from_selection(class_data)

                key = str(face) + "_" + str(frame.frame_id)
                cover[key] = np.array(fc, ndmin=2)

        return cover

    def collate_results(self, raw_ds):   
      """takes raw dictionaries with multiple observations per face (different frames) and collates them by face"""
        collated_ds = {}
        for obs in raw_ds:
            m = parse_cover_key.search(obs)
            face = int(m.group(1))
            if face in collated_ds:
                collated_ds[face] = np.append(collated_ds[face], np.array(raw_ds[obs],ndmin=2), axis=0)
            else:
                collated_ds[face] = np.array(raw_ds[obs], ndmin=2)

        return collated_ds

    def from_frame_interval(self, start=0,stop=0):
    """ determines fractional on faces visible in frames specified in interval. accelerated with multiprocessing""" 
        frames_selection = self.frames[start:stop]

        if self.manager is None:
            self.manager = mp.Manager()
        cover_raw = self.manager.dict()  #this will hold fractional cover data a: key 'faceid_frame_id', value: list of fractional cover,
                                    # this structure is simpler to deal with in multiprocesing context and will be postproccesed later
        jobs = []
        for chunk in chunks(frames_selection, math.ceil(len(frames_selection) / self.n_workers)):
            j = mp.Process(target=self.process_frames_prediction,
                           args=(chunk, cover_raw))
            j.start()
            jobs.append(j)

        for j in jobs:
            j.join()

        cover = self.collate_results(cover_raw)

        cover_avg = {}  #face_id: averaged_fractional_cover (over all observations of face)
        for face in cover:
            cover_avg[face] = np.mean(cover[face], axis=0)
           # print('face: {},   cover_avg: {}'.format(face, cover_avg[face]))

        for face in cover_avg:
            color = int(255 * cover_avg[face][4])
            self.mesh.visual.face_colors[face] = np.array([255-color, 255, 255-color, 255], dtype=np.uint8)

        return cover_avg, self.mesh

    def from_all_frames(self):
      """ determines fractional on faces visible in all registered frames"""
        start = 0
        stop = len(self.frames)
        return self.from_frame_interval(start, stop)

    def write_labels(self, labels=None, file_name=None):
     """ writes out fractional cover by face data to text file""" 
        fout = open(file_name, 'w')
        for face in labels:
            fout.write('{:d}'.format(face))
            for elm in labels[face]:
                fout.write('\t{:f}'.format(elm))
            fout.write('\n')

        fout.close()

    def _color_faces_from_images(self, frames, face_colors,  image_folder, ext):
    """ colors (rgb) each visible face of the mesh with the average color in associated images whose poses
        are provided in frame and actual image are in image_folder - can either be rgb images or semantic segmentation.
        primarily for visualization. target for multiprocessing """
        for frame in frames:
            print('working on {}, id: {}'.format(frame.label, frame.frame_id))
            hits, aabbs = frame.project_from_tree(self.tree, descend=4)
            hits_refined = self.refine_hits(frame, hits)

            img_path = image_folder + frame.label + ext
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            for face in hits_refined:
                triangle = hits_refined[face]
                selection, tri_coords = self.extract_triangle_from_image(triangle, img)
                mask = np.zeros((selection.shape[0], selection.shape[1]), dtype=np.uint8)
                cv2.fillConvexPoly(mask, tri_coords, 255)
                color = cv2.mean(selection, mask)
                key = str(face) + "_" + str(frame.frame_id)
                #face_colors[key] = {'color': color[0:3], 'coords': tri_coords}
                face_colors[key] = color[0:3]

        return face_colors


    def color_faces_from_images_interval(self, start, stop, image_folder, ext): 
    """ frontend for _color_faces_from_images() - colors mesh faces from frames in (start, stop) interval. 
        breaks those frames into chunks and doles out to multiprocessing targets """
        frames_selection = self.frames[start:stop]

        if self.manager is None:
            self.manager = mp.Manager()

        face_colors_raw = self.manager.dict()  # this will hold face color data a: key 'faceid_frame_id', value: list of face colors,
        # this structure is simpler to deal with in multiprocesing context and will be postproccesed later
        jobs = []
        for chunk in chunks(frames_selection, math.ceil(len(frames_selection) / self.n_workers)):
            j = mp.Process(target=self._color_faces_from_images,
                           args=(chunk, face_colors_raw, image_folder, ext))
            j.start()
            jobs.append(j)

        for j in jobs:
            j.join()

        face_colors = self.collate_results(face_colors_raw)

        face_colors_avg = {}  # face_id: averaged_fractional_cover (over all observations of face)
        for face in face_colors:
            face_colors_avg[face] = np.mean(face_colors[face], axis=0)

        for face in face_colors_avg:
            colors = face_colors_avg[face]
            self.mesh.visual.face_colors[face] = np.array([colors[0], colors[1], colors[2], 255], dtype=np.uint8)

        fout = open('face_color_test.txt', 'w')
        for face in face_colors:
            fout.write('{:d}'.format(face))
            for c in face_colors_avg[face]:
                fout.write('\t{:d}'.format(int(c)))
            fout.write('\n')

        fout.close()

        return self.mesh

    def color_faces_from_images_all(self, image_folder, ext): 
    """ colors all mesh faces visible in frames based on images in image_folder """ 
        start = 0
        stop = len(self.frames)
        return self.color_faces_from_images_interval(start, stop, image_folder, ext)





