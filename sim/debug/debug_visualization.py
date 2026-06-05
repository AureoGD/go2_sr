import numpy as np
import mujoco

RED = [1, 0, 0, 1]
GREEN = [0, 1, 0, 1]
BLUE = [0, 0, 1, 1]
GREY = [0.75, 0.75, 0.75, 1]


class DebugVisualizer:

    def __init__(self, viewer):
        self.viewer = viewer
        self.colors = [RED, GREY, GREY, GREY, GREY, GREEN, BLUE, RED]

    def _add_sphere(self, scn, pos, radius, rgba):
        if scn.ngeom >= scn.maxgeom:
            return

        geom = scn.geoms[scn.ngeom]

        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, [radius, 0, 0], pos, np.eye(3).flatten(), rgba)

        scn.ngeom += 1

    def render(self, spheres):
        if self.viewer is None:
            return

        scn = self.viewer.user_scn
        scn.ngeom = 0

        for i in range(len(spheres)):
            s = spheres[i]
            self._add_sphere(scn, pos=s, radius=0.015, rgba=self.colors[i])
