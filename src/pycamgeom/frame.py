import numpy as np
from collections import deque
import itertools

def all_binary_permutations(n):
   return [list(map(int, seq)) for seq in itertools.product("01", repeat=n)]


class Frame:
    """ represents a image and its pose in world coordinates.
        works with Camera and AABBtree from this package (pycamgeom)
    """

    def __init__(self):
        self.frame_id = []
        self.label = []
        self.enabled = True
        self.camera = []  # camera that acquired the image
        self.camera_id = []
        self.P = []  # projection matrix
        self.Tcw = []  # world to camera transform
        self.Twc = []  # camera to world transform

    @classmethod
    def load_agisoft(cls, xml_data, cameras):
        """ load frame data from xml file in agisoft format """
        _frame = cls()
        _frame.frame_id = xml_data.attrib['id']
        _frame.label = xml_data.attrib['label']
        _frame.camera_id = xml_data.attrib[
            'sensor_id']  # in agisoft terminology a sensor is what's called a camera here
        _frame.camera = cameras[_frame.camera_id]
        _frame.enabled = xml_data.attrib['enabled']

        transform = xml_data.find('transform')
        tdata_raw = [float(elm) for elm in transform.text.split()]
        _frame.Twc = np.array(tdata_raw).reshape((4, 4))
        _frame.Tcw = np.linalg.inv(_frame.Twc)

        _frame.P = np.matmul(_frame.camera.K, _frame.Tcw[0:3, :])
        return _frame

    def project(self, x_world):
        """ project point in world coordinates into image.
        the point will be converted to homogenous coordinates if it's not already; cooperates with camera
        returns a boolean indicating whether the point projected into the image and the image coordinates (regardless of whether the point is in the image)"""

        if x_world.shape != (4, 1):
            x_world = np.append(x_world, 1.000)  # make homogeneous
            x_world = x_world.reshape((4, 1))
        x_cam = np.matmul(self.Tcw, x_world)
        return self.camera.project(x_cam)

    def project_pinhole(self, x_world):
        """ project point in world coordinates into image using pinhole camera model
        returns a boolean indicating whether the point projected into the image and the image coordinates (regardless of whether the point is in the image)"""

        if x_world.shape != (4, 1):
            x_world = np.append(x_world, 1.000)  # make homogeneous
            x_world = x_world.reshape((4, 1))

        x_img = np.matmul(self.P, x_world)
        z_pos = x_img[2] > 0
        x_img = np.array([x_img[0] / x_img[2], x_img[1] / x_img[2]])
        in_image = (0 < x_img[0] < self.camera.dim[0]) and (0 < x_img[1] < self.camera.dim[1])
        return in_image, x_img, z_pos

    def backproject(self, u, v, z):
        """ backproject pixel coordinates u, v into world coordinates at distance z  """
        x_cam = self.camera.backproject(u, v, z)
        x_cam = np.append(x_cam, 1.000)
        x_cam = x_cam.reshape((4, 1))
        x_world = np.matmul(self.Twc, x_cam)
        return x_world[0:3]

    def aabb_is_visible(self, bounds):
        """ determines if any portion of a 3D aabb bounding box (defined by it's lower and upper bounds)
             is visible in the frame"""
        corners = []
        for perm in all_binary_permutations(3):
            corner = [ax[i] for i, ax in zip(perm, bounds)]
            corner = np.array(corner)
            corners.append(corner)

        w = self.camera.dim[0]
        h = self.camera.dim[1]
        x_le_w = []
        x_gt_0 = []
        y_le_h = []
        y_gt_0 = []
        z_pos = []
        positions = []
        positions2 = []
        for corner in corners:
            valid2, pos2 = self.project(corner)
            valid, pos, z_valid = self.project_pinhole(
                corner)  # this seems safer but will not be valid for highly non-linear cameras
            if z_valid:  # this is critical, if ignored projected image locations are wacky b/c a negative z can make points outside of image magically project in
                x_le_w.append((pos[0] < w))
                x_gt_0.append((pos[0] > 0))
                y_le_h.append((pos[1] < h))
                y_gt_0.append((pos[1] > 0))
            z_pos.append(z_valid)

            positions.append(pos)
            positions2.append(pos2)

        if sum(x_le_w) == 0 or sum(x_gt_0) == 0:
            return False  # box must be either entirely to left (sum(x_gt_0) == 0) or right (sum(x_le_w) == 0) of frame viewing area
        elif sum(y_le_h) == 0 or sum(y_gt_0) == 0:
            return False  # box must be entirely above or below viewing area
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
        for i in range(descend):  # if desired, descend into tree and start at lower level
            next_nodes = []
            for node in starting_nodes:
                next_nodes.extend([node.left, node.right])
            starting_nodes = next_nodes

        for node in starting_nodes:  # only append nodes that are visible to queue
            if self.aabb_is_visible(node.aabb.limits):
                queue.append(node)

        while queue:  # work until queue is empty to identify aabbs that hold potentially visible primitives
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
            pos = np.append(pos, np.expand_dims(xy, axis=0), axis=0)

        if (sum(valid) == 3):
            return True, pos
        else:
            return False, pos