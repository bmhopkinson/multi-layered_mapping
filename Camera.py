import numpy as np
from collections import deque
import itertools
import pdb


def all_binary_permutations(n):
   return [list(map(int, seq)) for seq in itertools.product("01", repeat=n)]


class Camera:
    """represents a projective camera using "OpenCV model", radial and tangential distortion """
    def __init__(self):
        self.id = []  
        self.dim = [] 	 #image dimensions
        self.K = []		# calibration matrix
        self.fx = []  	# focal lengths - x and y directions, pixels
        self.fy = []
        self.cx = []	# camera projection center - pixels
        self.cy = []
        self.k1 = 0.000   #radial distortion coeffs
        self.k2 = 0.000
        self.k3 = 0.000
        self.p1 = 0.000  # tangential distortion coeffs
        self.p2 = 0.000

    def project(self, x_cam):
        """ projects a homogenous point in camera coordinates into the camera image
            returns a boolean indicating whether the point projected into the image and the image coordinates (regardless of whether the point is in the image)"""
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
                            (self.p1 * (r**2 + 2 * x_inh[0]**2) + 2 * self.p2 * x_inh[0] * x_inh[1])

        yp =  x_inh[1] * (1 + self.k1 * r**2 + self.k2 * r**4 + self.k3 * r**6) + \
                            (self.p2 * (r**2 + 2 *  x_inh[1]**2) + 2 * self.p1 *  x_inh[0] * x_inh[1])
        x = self.cx + xp * self.fx
        y = self.cy + yp * self.fy

        in_image = (0 < x < self.dim[0]) and (0 < y <  self.dim[1])
        if in_image and sanity_check:
            return True, np.append(x,y)
        else:
            return False, np.append(x,y)

    def backproject(self, u, v, z):
        '''backproject points u,v (col, row) in pixel coordinates into 3D at distance z'''
        ud = (u - self.cx) / self.fx
        vd = (v - self.cy) / self.fy
        uc, vc = self.distortion_correction_oulu(ud, vd) # iteratively correct for radial distortion and tangential distortion
    #    pdb.set_trace()
        return np.array([uc*z, vc*z, z])


    def distortion_correction_oulu(self, u_raw, v_raw):
        '''for backprojection of pixel points. takes in distorted (real) focal length normalized coordinates in image (u/f, v/f) and
        corrects these points to where they would occur without distortion; modified from J-Y Bouguet Camera Calibration Toolbox for Matlab,
        based on Heikkila and Silven A four-step camera calibration procedure with implicit image correction 1997  '''
        N_ITER = 20

        k1 = self.k1 #radial distortion coeffs
        k2 = self.k2
        k3 = self.k3
        p1 = self.p1   # tangential distortion coeffs
        p2 = self.p2

        x = np.array([u_raw, v_raw]) # initial guess
        x_dist = x

        for i in range(N_ITER):
            r_2 = np.linalg.norm(x)
            k_radial = 1 + k1 * r_2 + k2 * r_2**2 + k3 * r_2**3
            dx1 = 2 * p1 * x[0]*x[1] + p2 * (r_2 + 2 * x[0]**2)
            dx2 = p1 * (r_2 + 2 * x[1]**2) + 2 * p2 * x[0] * x[1]
            delta_x = np.array([dx1, dx2])
            x = (x_dist - delta_x) / k_radial
         #   print('i: {}, x: {}'.format(i, x))

        return x[0], x[1]

    def load_agisoft(self, xml_data, version):
        """" load camera parameters from xml in agisoft format """

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
    """ represents a image and its pose in world coordinates"""

    def __init__(self):
        self.frame_id = []
        self.label = []
        self.enabled = True
        self.camera = []  # camera that acquired the image
        self.camera_id = []
        self.P = []  #projection matrix
        self.Tcw = []  # world to camera transform
        self.Twc = []   #camera to world transform

    def load_agisoft(self, xml_data, cameras):
        """ load frame data from xml file in agisoft format """
     
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
        """ project point in world coordinates into image.
        the point will be converted to homogenous coordinates if it's not already; cooperates with camera
        returns a boolean indicating whether the point projected into the image and the image coordinates (regardless of whether the point is in the image)"""

        if x_world.shape != (4,1):
            x_world = np.append(x_world, 1.000)  # make homogeneous
            x_world = x_world.reshape((4, 1))
        x_cam = np.matmul(self.Tcw, x_world)
        return self.camera.project(x_cam)

    def project_pinhole(self, x_world):
        """ project point in world coordinates into image using pinhole camera model
        returns a boolean indicating whether the point projected into the image and the image coordinates (regardless of whether the point is in the image)"""

        if x_world.shape != (4,1):
            x_world = np.append(x_world, 1.000)  # make homogeneous
            x_world = x_world.reshape((4, 1))

        x_img = np.matmul(self.P, x_world)
        z_pos = x_img[2] > 0
        x_img = np.array([x_img[0]/x_img[2], x_img[1]/x_img[2]])
        in_image = (0 < x_img[0] < self.camera.dim[0]) and (0 < x_img[1] < self.camera.dim[1])
        return in_image, x_img, z_pos

    def backproject(self, u, v, z):
        '''backproject pixel coordinates u, v into world coordinates at distance z'''
        x_cam = self.camera.backproject(u, v, z)
        x_cam = np.append(x_cam, 1.000)
        x_cam = x_cam.reshape((4, 1))
        x_world = np.matmul(self.Twc, x_cam)
        return x_world[0:3]

    def aabb_is_visible(self, bounds):
        """ determines if any portion of a 3D aabb bounding box (defined by it's lower and upper bounds)
             is visible in the frame"""
        #bound = list(zip(lb,ub))
        corners = []
        for perm in all_binary_permutations(3):
            corner = [ax[i] for i, ax in zip(perm, bounds)]
            corner = np.array(corner)
         #   corner = np.append(corner, 1.000)
         #   corner = corner.reshape((4,1))
            corners.append(corner)

        w = self.camera.dim[0]
        h = self.camera.dim[1]
        x_le_w =[]
        x_gt_0 = []
        y_le_h =[]
        y_gt_0 = []
        z_pos = []
        positions = []
        positions2 = []
        for corner in corners:
            valid2, pos2 = self.project(corner)
            valid, pos, z_valid = self.project_pinhole(corner)  #this seems safer but will not be valid for highly non-linear cameras
            if z_valid:  #this is critical, if ignored projected image locations are wacky b/c a negative z can make points outside of image magically project in
                x_le_w.append( (pos[0] < w) )
                x_gt_0.append( (pos[0] > 0) )
                y_le_h.append( (pos[1] < h) )
                y_gt_0.append( (pos[1] > 0) )
            z_pos.append(z_valid)

            positions.append(pos)
            positions2.append(pos2)

        if sum(x_le_w) == 0 or sum(x_gt_0) == 0:
            return False                            #box must be either entirely to left (sum(x_gt_0) == 0) or right (sum(x_le_w) == 0) of frame viewing area
        elif sum(y_le_h) == 0 or sum(y_gt_0) == 0:
            return False                           #box must be entirely above or below viewing area
        else:
            return True

    def project_from_tree(self, tree, descend=0):
        """ finds primitives in tree potentially visible in frame. returns primitives in leaf nodes of aabbtree for which some portion of box is visible in frame
        allow descent into the tree because i've found the frame transformation matrices from hyslam are perfectly fine locally but can have issues with global projection
        resulting in errors - make the process more local by descending into the tree returns indices of primitives potentially visible and associated aabb boxes"""
        hits = []
        aabbs = []
        queue = deque()

        starting_nodes = [tree]
        for i in range(descend):  #if desired, descend into tree and start at lower level
            next_nodes = []
            for node in starting_nodes:
                next_nodes.extend([node.left, node.right])
            starting_nodes = next_nodes

        for node in starting_nodes:  #only append nodes that are visible to queue
            if self.aabb_is_visible(node.aabb.limits):
                queue.append(node)

        while queue:   #work until queue is empty to identify aabbs that hold potentially visible primitives
            node = queue.popleft()

            if node.primitives:
                hits.extend(node.primitives)
                aabbs.append(node.aabb)

            for child in [node.left, node.right]:
                if child is not None:
                    if self.aabb_is_visible(child.aabb.limits):
                        queue.append(child)

        return hits, aabbs

    def project_triface(self, face_vertices):  
        """projects a triangular face into an image.
        returns a boolean indicating if the entire face is visible in the image and the associated position of the
        face vertices in the image"""
        
        valid = []
        pos = np.empty((0, 2))
        for vertex in face_vertices:
            val, xy = self.project(vertex)
            valid.append(val)
            #pos.append(xy)
            pos = np.append(pos, np.expand_dims(xy, axis=0), axis=0)

        if(sum(valid) ==3):
            return True, pos
        else:
            return False, pos











