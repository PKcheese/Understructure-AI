import numpy as np
from collections import deque

# ---- MediaPipe indices (subset you’ll need often) ----
NOSE=0; L_EYE_IN=1; R_EYE_IN=4
L_SHOULDER=11; R_SHOULDER=12; L_ELBOW=13; R_ELBOW=14; L_WRIST=15; R_WRIST=16
L_HIP=23; R_HIP=24; L_KNEE=25; R_KNEE=26; L_ANKLE=27; R_ANKLE=28; L_FOOT=31; R_FOOT=32

# Pairs defining the kinematic tree (parent -> child)
BONES = [
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP)
]

# Reasonable joint limits in degrees (min, max) for the main hinge joints.
# (Flexion only; you can expand with ab/adduction & twist per your needs.)
JOINT_LIMITS = {
    ("L_KNEE", (L_HIP, L_KNEE, L_ANKLE)): (0.0, 150.0),
    ("R_KNEE", (R_HIP, R_KNEE, R_ANKLE)): (0.0, 150.0),
    ("L_ELBOW", (L_SHOULDER, L_ELBOW, L_WRIST)): (0.0, 150.0),
    ("R_ELBOW", (R_SHOULDER, R_ELBOW, R_WRIST)): (0.0, 150.0),
}

# Normalised fallback lengths (after shoulder-width scaling).
DEFAULT_BONE_LENGTHS = {
    (L_SHOULDER, L_ELBOW): 0.36,
    (L_ELBOW, L_WRIST): 0.32,
    (R_SHOULDER, R_ELBOW): 0.36,
    (R_ELBOW, R_WRIST): 0.32,
    (L_HIP, L_KNEE): 0.45,
    (L_KNEE, L_ANKLE): 0.43,
    (R_HIP, R_KNEE): 0.45,
    (R_KNEE, R_ANKLE): 0.43,
    (L_SHOULDER, R_SHOULDER): 1.0,
    (L_HIP, R_HIP): 0.9,
}
MIN_SHOULDER_WIDTH = 1e-2
MIN_BONE_LENGTH = 5e-2
EPS = 1e-8

def vec(a,b):
    return b - a

def unit(v, eps=1e-8):
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n

def angle(a,b,c):
    """Angle at b from ba to bc in radians."""
    ba = unit(a - b); bc = unit(c - b)
    dot = np.clip(np.dot(ba, bc), -1.0, 1.0)
    return np.arccos(dot)

def clamp_angle(theta, deg_min, deg_max):
    return np.deg2rad(np.clip(np.rad2deg(theta), deg_min, deg_max))

def rotate_in_plane(a, b, c, target_theta):
    """
    Given points a-b-c (joint at b), rotate c around axis perpendicular to plane (a,b,c)
    to reach target angle at b while preserving |bc|.
    """
    # plane basis
    n = unit(np.cross(a-b, c-b))
    # current vectors
    v1 = unit(a-b); v2 = unit(c-b)
    theta = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    delta = target_theta - theta
    if abs(delta) < 1e-4:
        return c
    # Rodrigues rotation of v2 around n by delta
    k = n; v = v2
    v_rot = v*np.cos(delta) + np.cross(k, v)*np.sin(delta) + k*np.dot(k, v)*(1-np.cos(delta))
    length = np.linalg.norm(c-b)
    return b + v_rot*length

def pelvis_center(lms):
    return 0.5*(lms[L_HIP] + lms[R_HIP])

def shoulder_center(lms):
    return 0.5*(lms[L_SHOULDER] + lms[R_SHOULDER])

class PoseStabilizer:
    def __init__(self, ema_alpha=0.3):
        self.ema_alpha = ema_alpha
        self.prev = None
        self.calib_lengths = None  # dict { (i,j): length }
        self.history = deque(maxlen=3)  # for small temporal refs

    # --- 1) Normalize coords (origin pelvis, scale by shoulder width) ---
    def normalize(self, lms):
        lms = lms.copy()
        origin = pelvis_center(lms)
        lms -= origin
        # positive Z away from camera; MediaPipe uses z ~ negative toward camera.
        # flip sign to a conventional right-handed camera -> body space if you like:
        lms[:,2] *= -1.0

        sh_w = float(np.linalg.norm(lms[L_SHOULDER] - lms[R_SHOULDER]))
        if not np.isfinite(sh_w) or sh_w < MIN_SHOULDER_WIDTH:
            sh_w = 1.0
        scale = 1.0 / sh_w
        lms *= scale
        return lms, origin, sh_w

    def denormalize(self, lms, origin, shoulder_width):
        """Undo normalize(): rescale, unflip depth, and reapply origin."""
        restored = lms.copy()
        restored *= shoulder_width
        restored[:,2] *= -1.0
        restored += origin
        return restored

    # --- 2) Save bone lengths from a decent calibration frame (first stable frame) ---
    def maybe_calibrate(self, lms):
        if self.calib_lengths is not None:
            return
        self.calib_lengths = {}
        for i,j in BONES:
            length = float(np.linalg.norm(lms[j]-lms[i]))
            if not np.isfinite(length) or length < MIN_BONE_LENGTH:
                length = DEFAULT_BONE_LENGTHS.get((i,j)) or DEFAULT_BONE_LENGTHS.get((j,i)) or 0.5
            self.calib_lengths[(i,j)] = length

    # --- 3) Simple depth flip correction for elbows/knees ---
    def flip_correction(self, lms):
        # reference: limb should bend so that (upper->lower) lies on same side as the camera-facing normal
        # Use torso forward as reference direction (shoulders -> hips cross).
        up = unit(shoulder_center(lms) - pelvis_center(lms))
        lat = unit(lms[R_SHOULDER] - lms[L_SHOULDER])
        fwd = unit(np.cross(up, lat))  # approximate facing
        for name, (p, m, d) in JOINT_LIMITS.values():
            pass  # placeholder (dict values used only for tuple unpacking)
        # Elbows
        for (s,e,w) in [(L_SHOULDER,L_ELBOW,L_WRIST),(R_SHOULDER,R_ELBOW,R_WRIST),
                        (L_HIP,L_KNEE,L_ANKLE),(R_HIP,R_KNEE,R_ANKLE)]:
            v = vec(lms[e], lms[w])
            # if child points "behind" the expected facing for a typical curl, push it forward
            if np.dot(v, fwd) < -0.05:
                lms[w,2] = -lms[w,2]  # depth flip of the distal joint (simple but effective)
        return lms

    # --- 4) Enforce joint hinge limits (elbows/knees) ---
    def apply_joint_limits(self, lms):
        for name, triplet in JOINT_LIMITS.items():
            pass
        for key, lims in JOINT_LIMITS.items():
            _, (a,b,c) = key, lims  # just unpacking structure
        # Reformat dict to iterate cleanly
        limits = {
            (L_HIP, L_KNEE, L_ANKLE): JOINT_LIMITS["L_KNEE", (L_HIP, L_KNEE, L_ANKLE)],
            (R_HIP, R_KNEE, R_ANKLE): JOINT_LIMITS["R_KNEE", (R_HIP, R_KNEE, R_ANKLE)],
            (L_SHOULDER, L_ELBOW, L_WRIST): JOINT_LIMITS["L_ELBOW", (L_SHOULDER, L_ELBOW, L_WRIST)],
            (R_SHOULDER, R_ELBOW, R_WRIST): JOINT_LIMITS["R_ELBOW", (R_SHOULDER, R_ELBOW, R_WRIST)],
        }
        for (a,b,c), (amin, amax) in limits.items():
            th = angle(lms[a], lms[b], lms[c])
            th_clamped = clamp_angle(th, amin, amax)
            if abs(th - th_clamped) > 1e-3:
                lms[c] = rotate_in_plane(lms[a], lms[b], lms[c], th_clamped)
        return lms

    # --- 5) Lock bone lengths to calibration (one-step projection) ---
    def enforce_lengths(self, lms):
        if self.calib_lengths is None:
            return lms
        for i,j in BONES:
            target = self.calib_lengths.get((i,j))
            if target is None or target <= EPS:
                continue
            dir_ij = unit(lms[j]-lms[i])
            if not np.any(dir_ij):
                continue
            lms[j] = lms[i] + dir_ij*target
        return lms

    # --- 6) EMA smoothing over time ---
    def smooth(self, lms):
        if self.prev is None:
            self.prev = lms
            return lms
        a = self.ema_alpha
        sm = a*lms + (1-a)*self.prev
        self.prev = sm
        return sm

    # ---- Public entry point ----
    def stabilize(self, raw_landmarks_xyz, *, return_world=False):
        """
        raw_landmarks_xyz: np.ndarray (33,3) in MediaPipe coords
        returns: stabilized np.ndarray (33,3). By default this is pelvis-centered,
        scaled coordinates; set ``return_world=True`` to project back to
        MediaPipe/world coordinates.
        """
        raw_landmarks_xyz = np.asarray(raw_landmarks_xyz, dtype=np.float32)
        if raw_landmarks_xyz.shape != (33, 3):
            raise ValueError("Expected (33,3) landmark array.")
        if not np.all(np.isfinite(raw_landmarks_xyz)):
            return raw_landmarks_xyz.copy()

        lms_norm, origin, shoulder_width = self.normalize(raw_landmarks_xyz)
        lms = lms_norm
        self.maybe_calibrate(lms)
        steps = (
            self.flip_correction,
            self.apply_joint_limits,
            self.enforce_lengths,
            self.smooth,
        )
        for step in steps:
            lms = step(lms)
            if not np.all(np.isfinite(lms)):
                # revert to the best-known stable state (normalized input)
                lms = lms_norm
                break
        if return_world:
            return self.denormalize(lms, origin, shoulder_width)
        return lms


# ---- Convenience wrapper ----
_stabilizer = PoseStabilizer(ema_alpha=0.35)

def stabilize_landmarks(raw_landmarks_xyz: np.ndarray, *, return_world: bool = False) -> np.ndarray:
    """
    Call this once per frame with MediaPipe's 33x3 array (x,y,z).
    Returns stabilized landmarks (pelvis-centered, scaled) unless
    ``return_world=True`` is provided, in which case the coordinates are mapped
    back into the original MediaPipe/world space.
    """
    return _stabilizer.stabilize(raw_landmarks_xyz, return_world=return_world)
