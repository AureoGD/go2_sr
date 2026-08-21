import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

RED = [1, 0, 0, 1]
GREEN = [0, 1, 0, 1]
BLUE = [0, 0, 1, 1]
GREY = [0.75, 0.75, 0.75, 1]


class DebugVisualizer:

    def __init__(self, viewer):
        self.viewer = viewer
        self.colors = [RED, BLUE, GREY, GREEN, GREY, GREEN, BLUE, RED]

    def _add_sphere(self, scn, pos, radius, rgba):
        if scn.ngeom >= scn.maxgeom:
            return

        geom = scn.geoms[scn.ngeom]

        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0], pos, np.eye(3).flatten(), rgba)

        scn.ngeom += 1

    def _add_plane(self, scn, pos, rpy, rgba, size=(1.0, 1.0)):
        if scn.ngeom >= scn.maxgeom:
            return

        geom = scn.geoms[scn.ngeom]

        # Roll-pitch-yaw -> rotation matrix
        rot = R.from_euler('xyz', rpy).as_matrix()
        rot_plane = R.from_euler('xyz', np.array([0, -5 * np.pi / 180, 0])).as_matrix()
        rot = rot @ rot_plane

        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_PLANE, np.array([size[0], size[1], 0.01]), pos, rot.flatten(),
                            rgba)

        scn.ngeom += 1



    def render(self, spheres, plane_pos=None, rpy=None):
        if self.viewer is None:
            return

        scn = self.viewer.user_scn
        scn.ngeom = 0

        for i in range(len(spheres)):
            s = spheres[i]
            self._add_sphere(scn, pos=s, radius=0.015, rgba=self.colors[i])

        # self._add_plane(scn, plane_pos.flatten(), rpy, RED)
