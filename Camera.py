import numpy as np
import itertools
import xml.etree.ElementTree as ET
from collections import deque
import itertools


def all_binary_permutations(n):
   return [list(map(int, seq)) for seq in itertools.product("01", repeat=n)]


class Camera:
    def __init__(self):
        self.id = []
        self.dim = []
        self.K = []
        self.fx = []
        self.fy = []
        self.cx = []
        self.cy = []
        self.k1 = 0.000
        self.k2 = 0.000
        self.k3 = 0.000
        self.p1 = 0.000
        self.p2 = 0.000

    def project(self, x_cam):
        #pinhole projection for sanity check
        x_pinhole = np.matmul(self.K, x_cam[0:3,:])
        x_pinhole = np.array([x_pinhole[0]/x_pinhole[2], x_pinhole[1]/x_pinhole[2]])

        BUFFER = 0.5
        z_pos = (x_cam[2] > 0)
        in_image_pinhole = (-1*BUFFER*self.dim[0] < x_pinhole[0] < (1+BUFFER)*self.dim[0]) and (-1*BUFFER*self.dim[1]  < x_pinhole[1] < (1+BUFFER)*self.dim[1])
        sanity_check = False
        if z_pos and in_image_pinhole:  # point must be in front of camera
            sanity_check = True

        #distortion corrections
        x_inh = [x_cam[0] / x_cam[2], x_cam[1] / x_cam[2]]
        r = np.linalg.norm(np.array(x_inh))  # radial distance for distortion corrections

        xp = x_inh[0]  * (1 + self.k1 * r**2 + self.k2 * r**4 + self.k3 * r**6) + \
                            (self.p1 * (r**2 + 2 * x_inh[0]**2) + 2 * self.p2 * x_inh[0] * x_inh[1]);

        yp =  x_inh[1] * (1 + self.k1 * r**2 + self.k2 * r**4 + self.k3 * r**6) + \
                            (self.p2 * (r**2 + 2 *  x_inh[1]**2) + 2 * self.p1 *  x_inh[0] * x_inh[1]);
        x = self.cx + xp * self.fx;
        y = self.cy + yp * self.fy;

        in_image = (0 < x < self.dim[0]) and (0 < y <  self.dim[1])
        if(in_image and in_image_pinhole):
            return True, [x,y]
        else:
            return False, [x,y]

    def load_agisoft(self, xml_data, version):

        self.id = xml_data.attrib['id']

        # extract image dimensions
        dim = xml_data.find('resolution')
        self.dim = [float(dim.attrib['width']), float(dim.attrib['height'])]

        # process calibration, distortion parameters are present in variable numbers
        calib = xml_data.find('calibration')
        K = np.eye(3, dtype=float)

        fx = []
        fy = []
        if version == '1.4.0':
            fx = float(calib.find('f').text)
            fy = fx
        else:
            fx = float(calib.find('fx').text)
            fy = float(calib.find('fy').text)

        K[0, 0] = fx
        K[1, 1] = fy
        self.fx = fx
        self.fy = fy

        cx = float(calib.find('cx').text)
        cy = float(calib.find('cy').text)
        if version == '1.4.0':
            cx = cx + 0.5 * self.dim[0]
            cy = cy + 0.5 * self.dim[1]

        K[0, 2] = cx
        K[1, 2] = cy
        self.cx = cx
        self.cy = cy
        self.K = K

        # process distortion parameters, first radial distorion parameters (ks), then tangential (ps)
        k1 = calib.find('k1')
        if not k1 is None:
            self.k1 = float(k1.text)

        k2 = calib.find('k2')
        if not k2 is None:
            self.k2 = float(k2.text)

        k3 = calib.find('k3')
        if not k3 is None:
            self.k3 = float(k3.text)

        p1 = calib.find('p1')
        if not p1 is None:
            self.p1 = float(p1.text)

        p2 = calib.find('p2')
        if not p2 is None:
            self.p2 = float(p2.text)

class Frame:
    def __init__(self):
        self.frame_id = []
        self.label = []
        self.enabled = True
        self.camera = []
        self.camera_id = []
        self.P = []  #projection matrix
        self.Tcw = []  # world to camera transform
        self.Twc = []   #camera to world transform

    def load_agisoft(self, xml_data, cameras):
        self.frame_id = xml_data.attrib['id']
        self.label = xml_data.attrib['label']
        self.camera_id = xml_data.attrib['sensor_id']  #in agisoft terminology a sensor is what's called a camera here
        self.camera = cameras[self.camera_id]
        self.enabled = xml_data.attrib['enabled']

        transform = xml_data.find('transform')
        tdata_raw = [float(elm) for elm in transform.text.split()]
        self.Twc = np.array(tdata_raw).reshape((4,4))
        self.Tcw = np.linalg.inv(self.Twc)

        self.P = np.matmul(self.camera.K, self.Tcw[0:3, :])



    def project(self, x_world):
        x_cam = np.matmul(self.Tcw, x_world)
        return self.camera.project(x_cam)

    def project_pinhole(self, x_world):
        x_img = np.matmul(self.P, x_world)
        z_pos = x_img[2] > 0
        x_img = [x_img[0]/x_img[2], x_img[1]/x_img[2]]
        in_image = (0 < x_img[0] < self.camera.dim[0]) and (0 < x_img[1] < self.camera.dim[1])
        return in_image, x_img, z_pos


    def aabb_is_visible(self, bounds):
        #bound = list(zip(lb,ub))
        corners = []
        for perm in all_binary_permutations(3):
            corner = [ax[i] for i, ax in zip(perm, bounds)]
            corner = np.array(corner)
            corner = np.append(corner, 1.000)
            corner = corner.reshape((4,1))
            corners.append(corner)

        w = self.camera.dim[0]
        h = self.camera.dim[1]
        x_le_w =[]
        x_gt_0 = []
        y_le_h =[]
        y_gt_0 = []
        z_pos = []
        positions = []

        for corner in corners:
            #valid, pos = self.project(corner)
            valid, pos, z_valid = self.project_pinhole(corner)
            x_le_w.append( (pos[0] < w) )
            x_gt_0.append( (pos[0] > 0) )
            y_le_h.append( (pos[1] < h) )
            y_gt_0.append( (pos[1] > 0) )
            z_pos.append(z_valid)

            positions.append(pos)
       #     positions2.append(pos2)

        if sum(x_le_w) == 0 or sum(x_gt_0) == 0:
            return False                            #box must be either entirely to left (sum(x_gt_0) == 0) or right (sum(x_le_w) == 0) of frame viewing area
        elif sum(y_le_h) == 0 or sum(y_gt_0) == 0:
            return False                           #box must be entirely above or below viewing area
        elif sum(z_pos) == 0:
            return False
        else:
            return True

    def project_from_tree(self, tree):
        #finds primitives in tree potentially visible in frame. returns primitives in leaf nodes of aabbtree for which some portion of box is visible in frame
        hits = []
        queue = deque()
        # probably should double check root.aabb hits before appending

        if self.aabb_is_visible(tree.aabb.limits):
            queue.append(tree)

        while queue:
            node = queue.popleft()

            if node.primitives:
                hits.extend(node.primitives)

            for child in [node.left, node.right]:
                if child is not None:
                    if self.aabb_is_visible(child.aabb.limits):
                        queue.append(child)

        return hits









