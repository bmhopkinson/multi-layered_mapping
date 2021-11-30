import numpy as np

class Camera:
    """represents a projective camera using "OpenCV model", radial and tangential distortion """
    def __init__(self):
        self.id = []
        self.dim = [] 	# image dimensions
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

    @classmethod
    def load_agisoft(cls, xml_data, version):
        """" load camera parameters from xml in agisoft format """

        _camera = cls()
        _camera.id = xml_data.attrib['id']

        # extract image dimensions
        dim = xml_data.find('resolution')
        _camera.dim = [float(dim.attrib['width']), float(dim.attrib['height'])]

        # process calibration, distortion parameters are present in variable numbers
        calib = xml_data.find('calibration')
        K = np.eye(3, dtype=float)

        if version == '1.4.0':
            fx = float(calib.find('f').text)
            fy = fx
        else:
            fx = float(calib.find('fx').text)
            fy = float(calib.find('fy').text)

        K[0, 0] = fx
        K[1, 1] = fy
        _camera.fx = fx
        _camera.fy = fy

        cx = float(calib.find('cx').text)
        cy = float(calib.find('cy').text)
        if version == '1.4.0':
            cx = cx + 0.5 * _camera.dim[0]
            cy = cy + 0.5 * _camera.dim[1]

        K[0, 2] = cx
        K[1, 2] = cy
        _camera.cx = cx
        _camera.cy = cy
        _camera.K = K

        # process distortion parameters, first radial distorion parameters (ks), then tangential (ps)
        k1 = calib.find('k1')
        if k1 is not None:
            _camera.k1 = float(k1.text)

        k2 = calib.find('k2')
        if k2 is not None:
            _camera.k2 = float(k2.text)

        k3 = calib.find('k3')
        if k3 is not None:
            _camera.k3 = float(k3.text)

        p1 = calib.find('p1')
        if p1 is not None:
            _camera.p1 = float(p1.text)

        p2 = calib.find('p2')
        if p2 is not None:
            _camera.p2 = float(p2.text)

        return _camera

    def project(self, x_cam):
        """ projects a homogenous point in camera coordinates into the camera image
            returns a boolean indicating whether the point projected into the image and the image coordinates (regardless of whether the point is in the image)"""
        # pinhole projection for sanity check
        x_pinhole = np.matmul(self.K, x_cam[0:3, :])
        x_pinhole = np.array([x_pinhole[0]/x_pinhole[2], x_pinhole[1]/x_pinhole[2]])

        BUFFER = 0.5
        z_pos = (x_cam[2] > 0)
        in_image_pinhole = (-1*BUFFER*self.dim[0] < x_pinhole[0] < (1+BUFFER)*self.dim[0]) and (-1*BUFFER*self.dim[1]  < x_pinhole[1] < (1+BUFFER)*self.dim[1])
        sanity_check = False
        if z_pos and in_image_pinhole:  # point must be in front of camera
            sanity_check = True

        # distortion corrections
        x_inh = [x_cam[0] / x_cam[2], x_cam[1] / x_cam[2]]
        r = np.linalg.norm(np.array(x_inh))  # radial distance for distortion corrections

        xp = x_inh[0] * (1 + self.k1 * r**2 + self.k2 * r**4 + self.k3 * r**6) + \
                        (self.p1 * (r**2 + 2 * x_inh[0]**2) + 2 * self.p2 * x_inh[0] * x_inh[1])

        yp =  x_inh[1] * (1 + self.k1 * r**2 + self.k2 * r**4 + self.k3 * r**6) + \
                         (self.p2 * (r**2 + 2 * x_inh[1]**2) + 2 * self.p1 * x_inh[0] * x_inh[1])
        x = self.cx + xp * self.fx
        y = self.cy + yp * self.fy

        in_image = (0 < x < self.dim[0]) and (0 < y < self.dim[1])
        if in_image and sanity_check:
            return True, np.append(x, y)
        else:
            return False, np.append(x, y)

    def backproject(self, u, v, z):
        """ backproject points u,v (col, row) in pixel coordinates into 3D at distance z"""
        ud = (u - self.cx) / self.fx
        vd = (v - self.cy) / self.fy
        uc, vc = self.distortion_correction_oulu(ud, vd)  # iteratively correct for radial distortion and tangential distortion
    #    pdb.set_trace()
        return np.array([uc*z, vc*z, z])


    def distortion_correction_oulu(self, u_raw, v_raw):
        """ for backprojection of pixel points. takes in distorted (real) focal length normalized coordinates in image (u/f, v/f) and
        corrects these points to where they would occur without distortion; modified from J-Y Bouguet Camera Calibration Toolbox for Matlab,
        based on Heikkila and Silven A four-step camera calibration procedure with implicit image correction 1997  """
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

        return x[0], x[1]