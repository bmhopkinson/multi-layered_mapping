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
    def __init__(self, frames=None, mesh=None, tree=None, img_dir=[], n_workers=1):
        self.frames = frames
        self.mesh = copy.deepcopy(mesh)
        self.vertices = []
        self.faces = []
        self.tree = tree
        self.img_dir = img_dir
        self.n_workers = n_workers

        if self.mesh is not None:
            self.vertices = mesh.vertices.view(np.ndarray)
            self.faces = mesh.faces.view(np.ndarray)

    def refine_hits(self, frame, hits):
        hits_refined = {}
        for hit in hits:
            vertex_ids = self.faces[hit, :]
            face_vertices = self.vertices[vertex_ids, :]
            valid, pos = frame.project_triface(face_vertices)
            if valid:
                hits_refined[hit] = pos

        return hits_refined

    def extract_triangle_from_prediction_image(self, triangle, image):
        # select only the portion of  prediction image within this triangle and covert rgb codes to class data
        triangle.append(triangle[0]) #close triangle
        triangle = np.array(triangle, dtype=int)
        mask = np.zeros((image.shape[0], image.shape[1]))

        #create mask over triangular region
        cv2.fillConvexPoly(mask, triangle, 1)
        mask = mask.astype(np.bool)
        selection = np.zeros_like(image)
        selection[mask] = image[mask]  #extract predictions within mask region

        # crop for faster processing
        xy_min = np.amin(triangle, axis=0)
        xy_max = np.amax(triangle, axis=0)
        selection_crop = selection[xy_min[1]:xy_max[1], xy_min[0]:xy_max[0]]

        return selection_crop

    def maskrgb_to_class(self, mask):
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
        pixel_count = []
        for i in range(n_classes):
            t = np.sum(class_data == i)
            pixel_count.append(t)

        return pixel_count / np.sum(pixel_count)

    def process_frames(self, frames, cover):
        for frame in frames:
            print('working on {}, id: {}'.format(frame.label, frame.frame_id))
            hits, aabbs = frame.project_from_tree(self.tree, descend=4)
            hits_refined = self.refine_hits(frame,hits)

            img_pred_path = self.img_dir + frame.label + "_pred.png"
            img_pred = cv2.imread(img_pred_path)
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

            for face in hits_refined:
                triangle = hits_refined[face]
                selection = self.extract_triangle_from_prediction_image(triangle, img_pred)
                class_data = self.maskrgb_to_class(selection)
                fc = self.fractional_cover_from_selection(class_data)

                key = str(face) + "_" + str(frame.frame_id)
                cover[key] = np.array(fc, ndmin=2)

                # if face in cover:
                #     cover[face] = np.append(cover[face], np.array(fc,ndmin=2), axis=0)
                # else:
                #     cover[face] = np.array(fc, ndmin=2)

        return cover

    def from_frame_interval(self, start=0,stop=0):
        frames_selection = self.frames[start:stop]

        manager = mp.Manager()
        cover_raw = manager.dict()  #this will hold fractional cover data a: key 'faceid_frame_id', value: list of fractional cover,
                                    # this structure is simpler to deal with in multiprocesing context and will be postproccesed later
        jobs = []
        for chunk in chunks(frames_selection, math.ceil(len(frames_selection) / self.n_workers)):
            j = mp.Process(target=self.process_frames,
                           args=(chunk, cover_raw))
            j.start()
            jobs.append(j)

        for j in jobs:
            j.join()

        cover = {}  # face_id: n_obs x n_class nparray holding fracional cover observation for obs_i, class_i
        for obs in cover_raw:
            m = parse_cover_key.search(obs)
            face = int(m.group(1))
            if face in cover:
                cover[face] = np.append(cover[face], np.array(cover_raw[obs],ndmin=2), axis=0)
            else:
                cover[face] = np.array(cover_raw[obs], ndmin=2)

        cover_avg = {}  #face_id: averaged_fractional_cover (over all observations of face)
        for face in cover:
            cover_avg[face] = np.mean(cover[face], axis=0)
           # print('face: {},   cover_avg: {}'.format(face, cover_avg[face]))

        for face in cover_avg:
            color = int(255 * cover_avg[face][4])
            self.mesh.visual.face_colors[face] = np.array([255-color, 255, 255-color, 255], dtype=np.uint8)

        return cover_avg, self.mesh

    def from_all_frames(self):
        start = 0
        stop = len(self.frames)
        return self.from_frame_interval(start, stop)

    def write_labels(self, labels=None, file_name=None):
        fout = open(file_name, 'w')
        for face in labels:
            fout.write('{:d}'.format(face))
            for elm in labels[face]:
                fout.write('\t{:f}'.format(elm))
            fout.write('\n')

        fout.close()



