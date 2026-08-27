# Cameras and calibration

Every camera trajectory (rigid pose *and* intrinsics, per frame) lives in
`processed_data/metric_extrinsics/<camera>/<mode>.npz`, one file per
`(camera, mode)` pair, covering every frame of the sequence. This is the
**only** calibration source you should need -- `contact4d.io.load_camera`
wraps it into a `contact4d.cameras.Camera` you can hand to every other tool
in this package.

## World frame

All annotations (`poses3d`, `smpl`, `smplx`, `mano_params`, ...) live in one
shared **metric world frame**, anchored to one Aria device's trajectory
(see `processed_data/metric_extrinsics/_metadata.json`'s `"anchor_aria"`
field). Units are meters.

## `metric_extrinsics/<camera>/<mode>.npz` schema

| array | shape | dtype | meaning |
|---|---|---|---|
| `frame_ids` | (T,) | int32 | frame ids this trajectory covers |
| `world_to_camera` | (T,4,4) | float64 | rigid transform, world -> camera, per frame |
| `camera_to_world` | (T,4,4) | float64 | its inverse |
| `intrinsics` | (T,P) | float64 | P=8 for exo cameras, P=15 for Aria |
| `source_calibration_frame_ids` | (T,) | int32 | which calibration capture frame each entry's intrinsics came from |
| `image_width`, `image_height` | scalar | int32 | the **as-shipped** image size for this camera/mode |

Convention: **column vectors**, `x_camera = world_to_camera @ [x_world, y_world, z_world, 1]`.

`processed_data/metric_extrinsics/_metadata.json` (one per sequence) lists
every exported `(camera, mode)` pair plus, per stream, its `camera_model`
string (`"OPENCV_FISHEYE"` or `"ARIA_RADTAN_THIN_PRISM_FISHEYE_15"`) and
image size -- `contact4d.io.load_camera`/`MetricExtrinsics` read this for
you automatically.

## Camera models

Two intrinsics models are used across the dataset. Both are implemented
from scratch in pure numpy in `contact4d.cameras` -- **no external SDK is
required for either**, including the Aria one.

### Exo cameras: `exo_fisheye` (`OPENCV_FISHEYE`, 8 params)

`[fx, fy, cx, cy, k1, k2, k3, k4]` -- OpenCV's equidistant fisheye model
(https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html). Used by
every `camNN` stream (`mode="rgb"` only).

### Aria cameras: `aria_fisheye` (`ARIA_RADTAN_THIN_PRISM_FISHEYE_15`, 15 params)

`[f, cu, cv, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3]` -- one shared
focal length, 6 radial + 2 tangential + 4 thin-prism distortion terms. Used
by every `ariaNN` stream (`mode` one of `rgb`/`left`/`right`).

**Rotation quirk**: Aria's `left`/`right` (SLAM) streams are stored rotated
90 degrees from the camera's native sensor frame (`rgb` is square, so this
doesn't matter for it). `Camera.project`/`Camera.cam_to_image` already apply
this rotation automatically for `mode in ("left", "right")` -- see
`contact4d.cameras.aria_fisheye.rotate_to_upright` if you're projecting
manually instead of going through `Camera`.

## Using `Camera`

```python
from contact4d import io, projection

camera = io.load_camera(sequence_path, "cam05", "rgb", frame_id=1)
body_3d = io.load_annotation(io.annotation_path(sequence_path, "fit_poses3d", frame_id=1))
body_2d = projection.project_body_pose3d(body_3d, camera)  # {person_id: (17,3) x,y,conf}
```

`Camera.world_to_cam(points)` gives you camera-frame 3D (item 5);
`Camera.project(points)` chains that with `Camera.cam_to_image(...)` to give
pixel coordinates (item 1). `contact4d.undistort` uses the same `Camera` to
build a pinhole-undistorted view of either camera model (item 4).
