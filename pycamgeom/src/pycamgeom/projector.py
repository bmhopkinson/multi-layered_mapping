import numpy as np
import trimesh

class Projector():

    def __init__(self, faces, vertices, mesh, tree, descend=4):
        self.faces = faces
        self.vertices = vertices
        self.mesh = mesh
        self.tree = tree
        self.descend = descend
        self.ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self.mesh)

    def find_visible_faces(self, frame):
        hits, aabbs = frame.project_from_tree(self.tree, descend=self.descend)
        return self.refine_hits(frame, hits)

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

        if hits_refined:  # assuming we have potential visible faces, check for line of sight
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
                ray_dir = face_center - cam_center.reshape((1 ,3))
            else:
                ray_dir = cam_center.reshape((1, 3)) - face_center

            ray_dir = ray_dir /np.linalg.norm(ray_dir)
            ray_dirs = np.append(ray_dirs, ray_dir, axis=0)

            # ray origin
            if forward:
                ray_orgs = np.append(ray_orgs, cam_center.reshape((1, 3)), axis=0)
            else:
                # ray origin will be a point along the ray direction slightly off the face center
                v_deltas = face_vertices - face_vertices[[2, 0, 1], :]
                edge_len_mean = np.mean(np.linalg.norm(v_deltas, axis=1))
                ray_org = face_center + 0.05 *edge_len_mean *ray_dir
                ray_orgs = np.append(ray_orgs, ray_org, axis=0)

        # conduct intersection test
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