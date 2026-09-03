# Annotation reference

All per-frame annotation files under `processed_data/<kind>/` are `.npy`
files holding a single Python `dict`, saved as a 0-d numpy object array
(`np.load(path, allow_pickle=True).item()` -- or just use
`contact4d.io.load_annotation(path)`). They're keyed by person id
(e.g. `"aria01"`) for body annotations, or by hand side (`"left"`/`"right"`)
for hand annotations. `camera_params` is the one exception: it's one
`.npz` per camera/mode covering every frame of the sequence at once (see
`docs/cameras.md`).

All coordinates are in the sequence's shared **metric world frame** (meters)
unless noted otherwise ("camera space" annotations use each camera's own
frame instead -- see `docs/cameras.md` and `contact4d.camera_space`).

## Body 3D keypoints -- `body_pose3d/`

`{person_id: array(17, 4)}`, `float64`, columns `(x, y, z, confidence)`.
The temporally-smoothed/refined 3D body keypoints -- the recommended source
for downstream use.

## Hand 3D keypoints -- `hand_pose3d/`

`{"left"/"right": array(21, 4)}`, `float64`, columns `(x, y, z, confidence)`,
MANO joint order. This is the recommended (post-corrective) source for hand
3D keypoints.

## Body 2D keypoints -- `body_poses2d/<camera_or_aria>/<mode>/<frame>.npy`

`{person_id: array(17, 3)}`, `float64`, columns `(x, y, confidence)` in
pixel coordinates for that specific camera/mode. Matches the output shape of
`contact4d.projection.project_body_pose3d`.

## Hand 2D keypoints -- `hand_poses2d_corrective/<camera_or_aria>/<mode>/<frame>.npy`

`{"left"/"right": array(21, 3)}`, `float64`, columns `(x, y, confidence)`.
Matches the output shape of `contact4d.projection.project_hand_pose3d`.

## SMPL -- `smpl/`

`{person_id: {...}}`:

| key | shape | dtype | notes |
|---|---|---|---|
| `betas` | (10,) | f32 | shape coefficients |
| `global_orient` | (3,) | f32 | axis-angle |
| `body_pose` | (69,) | f32 | 23 joints x 3, axis-angle |
| `transl` | (3,) | f32 | |
| `vertices` | (6890, 3) | f32 | world space |
| `joints` | (45, 3) | f32 | world space; 24 SMPL + 21 extra joints |
| `epoch_loss` | scalar | float | fitting diagnostic |

## SMPL-X -- `smplx/`

`{person_id: {...}}`:

| key | shape | dtype | notes |
|---|---|---|---|
| `betas` | (10,) | f32 | |
| `global_orient` | (3,) | f32 | axis-angle |
| `body_pose` | (63,) | f32 | 21 joints x 3, axis-angle |
| `transl` | (3,) | f32 | |
| `left_hand_pose` | (45,) | f32 | 15 joints x 3, axis-angle |
| `right_hand_pose` | (45,) | f32 | |
| `vertices` | (10475, 3) | f32 | world space |
| `joints` | (127, 3) | f32 | world space |

## MANO -- `mano/` (`schema_version: 2`)

`{"left"/"right": {...}}`:

| key | shape | dtype | notes |
|---|---|---|---|
| `side` | str | | `"left"` or `"right"` |
| `betas` | (10,) | f32 | shape coefficients |
| `betas_source` | str | | path to the sequence's shared `mano_shape_betas.npy` |
| `phis` | (21, 2) | f32 | pose-space phase parameters |
| `pose3d` | (21, 4) | f64 | the 3D keypoint skeleton the fit targeted, (x,y,z,conf) |
| `global_orient` | (3, 3) | f32 | root rotation matrix |
| `hand_pose` | (15, 3, 3) | f32 | per-joint rotation matrices |
| `rot_mats` | (16, 3, 3) | f32 | `global_orient` + `hand_pose` stacked |
| `rot_quats` | (16, 4) | f32 | `rot_mats` as `(w, x, y, z)` quaternions |
| `transl` | (3,) | f32 | |
| `vertices` | (778, 3) | f32 | world space |
| `joints` | (21, 3) | f32 | world space |
| `vertices_root_centered` | (778, 3) | f32 | `vertices - transl` |
| `joints_root_centered` | (21, 3) | f32 | `joints - transl` |
| `root_joint` | (3,) | f32 | |
| `beta_loss`, `theta_loss` | scalar | | fitting diagnostics |

## Camera-space model params -- `cam_smpl/`, `cam_smplx/`, `cam_mano/`

`processed_data/cam_<model>/<camera>/<mode>/<frame>.npy` -- produced by
`contact4d.camera_space` (or, internally, the same annotations may already
be shipped for some sequences). Same keys as the corresponding world-space
annotation above, **plus**:

| key | notes |
|---|---|
| `camera_space_schema_version` | `1` |
| `coordinate_system` | `"camera_native_opencv_x_right_y_down_z_forward_meters"` |
| `world_to_camera` | (4,4) f64, the transform used |
| `canonical_pelvis` | (3,) f32 -- smpl/smplx only |
| `camera_space_validation` | dict of reconstruction-error diagnostics (see `contact4d.camera_space`) |

`global_orient`/`transl` (and, for MANO, `pose3d`/`rot_mats`/`rot_quats`/
`vertices`/`joints`/the root-centered fields) are all re-expressed in that
camera's frame; local pose (`body_pose`, `hand_pose`, MANO's per-joint
`hand_pose`) and shape (`betas`) are unchanged.

## Fingertip contact -- `finger_contact/annotations.json`

Ground-truth per-fingertip contact state -- Contact4D's core contribution
alongside pose. One JSON file per sequence (not one `.npy` per frame like
every other annotation type): `{"<frame_id>": {"left_thumb": bool,
"left_index": bool, "left_middle": bool, "left_ring": bool, "left_pinky":
bool, "right_thumb": bool, ..., "right_pinky": bool}}`. `True` means that
fingertip is in contact with an object/surface in that frame. Not every
frame in the sequence is necessarily annotated. Load with
`contact4d.io.load_finger_contact` (converts keys to `int` frame ids,
matching every other frame-id convention here); each finger name maps onto
the 21-joint MANO/`hand_pose3d` skeleton via
`contact4d.finger_contact.FINGERTIP_JOINTS`.
`contact4d.visualize.draw_finger_contact_2d` overlays contact state as two
fixed corner panels (one per hand, five dots each -- red = contact, green =
no contact) rather than on the hand itself, so it needs no keypoint
projection; see its docstring for the `mirror` option (on by default, for
an exo camera facing the subject).

## Camera calibration -- `camera_params/<camera>/<mode>.npz`

See `docs/cameras.md` for the full schema and both supported intrinsics models.

## Init-stage annotations -- `init_smpl/`, `init_smplx/`

Earlier-stage (pre-final-fit) versions of `smpl`/`smplx`, useful mainly for
debugging the fitting pipeline; most users want `smpl`/`smplx` instead.
