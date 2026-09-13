import argparse
import math
 
MESH_DIR = "../meshes"
BASE_Z_OFFSET   = 0.080         
BASE_BOX_HALF   = (0.15, 0.125, 0.05)   # URDF box size /2 (MuJoCo uses half-extents)
BASE_MASS       = 2.5
BASE_DIAGINERTIA = (0.020, 0.040, 0.050)
 
WHEEL_RADIUS = 0.0795
WHEEL_HALF_LEN = 0.0225           # URDF length 0.045 / 2
WHEEL_MASS = 0.25
WHEEL_DIAGINERTIA = (0.00043, 0.00079, 0.00043)
WHEEL_VEL_LIMIT = 30              # URDF <limit velocity="30">
 
WHEELS = {
    "front_left":  (0.30,  0.19, -0.08),
    "front_right": (0.30, -0.19, -0.08),
    "rear_left":   (-0.05, 0.19, -0.08),
    "rear_right":  (-0.05, -0.19, -0.08),
}
 
CAMERA_POS = (0.3, -0.04, -0.167)
LIDAR_POS  = (0.09, -0.02, 0.065)
N_RANGE_RAYS = 12
RANGE_MAX = 5.0
 
 
def wheel_body(name_suffix, wheel_name, pos):
    x, y, z = pos
    jname = f"{wheel_name}_joint{name_suffix}"
    return f"""
      <body name="{wheel_name}{name_suffix}" pos="{x} {y} {z}">
        <inertial pos="0 0 0" mass="{WHEEL_MASS}" diaginertia="{WHEEL_DIAGINERTIA[0]} {WHEEL_DIAGINERTIA[1]} {WHEEL_DIAGINERTIA[2]}"/>
        <joint name="{jname}" type="hinge" axis="0 1 0" limited="false"/>
        <geom type="mesh" mesh="{wheel_name}_mesh" class="visual"/>
        <geom type="cylinder" size="{WHEEL_RADIUS} {WHEEL_HALF_LEN}" euler="90 0 0"
              class="collision" friction="1.0 0.005 0.0001"/>
      </body>"""
 
 
def rangefinder_sites(name_suffix):
    sites = []
    sensors = []
    for i in range(N_RANGE_RAYS):
        ang = 360.0 * i / N_RANGE_RAYS
        sname = f"range{i}{name_suffix}"
        sites.append(
            f'        <site name="{sname}" pos="{LIDAR_POS[0]} {LIDAR_POS[1]} {LIDAR_POS[2]}" '
            f'euler="0 0 {ang}" size="0.005"/>'
        )
        sensors.append(f'    <rangefinder name="rf_{sname}" site="{sname}" cutoff="{RANGE_MAX}"/>')
    return "\n".join(sites), "\n".join(sensors)
 
 
def robot_body(index, pos):
    suf = f"_{index}"
    x, y = pos
    wheel_bodies = "\n".join(
        wheel_body(suf, wname, wpos) for wname, wpos in WHEELS.items()
    )
    range_sites, _ = rangefinder_sites(suf)
 
    return f"""
    <body name="piple{suf}" pos="{x} {y} {BASE_Z_OFFSET}">
      <freejoint name="root{suf}"/>
      <inertial pos="0 0 0" mass="{BASE_MASS}"
                diaginertia="{BASE_DIAGINERTIA[0]} {BASE_DIAGINERTIA[1]} {BASE_DIAGINERTIA[2]}"/>
 
      <geom type="mesh" mesh="piple_base_link_mesh" pos="0.02 -0.02 0" class="visual"/>
      <geom type="box" size="{BASE_BOX_HALF[0]} {BASE_BOX_HALF[1]} {BASE_BOX_HALF[2]}" class="collision"/>
 
      <site name="cam_site{suf}" pos="{CAMERA_POS[0]} {CAMERA_POS[1]} {CAMERA_POS[2]}" size="0.01"/>
      <camera name="cam{suf}" pos="{CAMERA_POS[0]} {CAMERA_POS[1]} {CAMERA_POS[2]}" mode="fixed" fovy="90"/>
 
      <site name="lidar_site{suf}" pos="{LIDAR_POS[0]} {LIDAR_POS[1]} {LIDAR_POS[2]}" size="0.01"/>
{range_sites}
 
{wheel_bodies}
    </body>"""
 
 
def robot_actuators(index):
    suf = f"_{index}"
    lines = []
    for wname in WHEELS:
        jname = f"{wname}_joint{suf}"
        lines.append(
            f'    <velocity name="act_{jname}" joint="{jname}" kv="8" '
            f'ctrlrange="-{WHEEL_VEL_LIMIT} {WHEEL_VEL_LIMIT}"/>'
        )
    return "\n".join(lines)
 
 
def robot_sensors(index):
    suf = f"_{index}"
    _, range_sensors = rangefinder_sites(suf)
    lines = [
        f'    <framepos name="pos{suf}" objtype="body" objname="piple{suf}"/>',
        f'    <framequat name="quat{suf}" objtype="body" objname="piple{suf}"/>',
        f'    <framelinvel name="linvel{suf}" objtype="body" objname="piple{suf}"/>',
        f'    <frameangvel name="angvel{suf}" objtype="body" objname="piple{suf}"/>',
        range_sensors,
    ]
    return "\n".join(lines)
 
 
def build(n, spacing, out_path):
    cols = math.ceil(math.sqrt(n))
    positions = []
    for i in range(n):
        row, col = divmod(i, cols)
        x = (col - (cols - 1) / 2.0) * spacing
        y = (row - (cols - 1) / 2.0) * spacing
        positions.append((x, y))
 
    bodies = "\n".join(robot_body(i, p) for i, p in enumerate(positions))
    actuators = "\n".join(robot_actuators(i) for i in range(n))
    sensors = "\n".join(robot_sensors(i) for i in range(n))
 
    xml = f"""<mujoco model="piple_arena">
 
  <compiler angle="degree" meshdir="{MESH_DIR}" autolimits="true"/>
  <option timestep="0.005" integrator="implicitfast"/>
 
  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/>
    <rgba haze="0.55 0.65 0.75 1"/>
    <global offwidth="1280" offheight="720"/>
  </visual>
 
  <asset>
    <mesh name="piple_base_link_mesh" file="piple_base_link.obj"/>
    <mesh name="front_left_wheel_mesh" file="front_left_wheel_link.obj"/>
    <mesh name="front_right_wheel_mesh" file="front_right_wheel_link.obj"/>
    <mesh name="rear_left_wheel_mesh" file="rear_left_wheel_link.obj"/>
    <mesh name="rear_right_wheel_mesh" file="rear_right_wheel_link.obj"/>
    <mesh name="camera_mesh" file="camera.obj"/>
 
    <texture type="skybox" builtin="gradient" rgb1="0.65 0.78 0.90" rgb2="1 1 1" width="256" height="256"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.78 0.78 0.78" rgb2="0.58 0.58 0.58"
              width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="30 30" reflectance="0.05" specular="0.1" shininess="0.05"/>
  </asset>
 
  <default>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="2" rgba="0.75 0.75 0.78 1"/>
    </default>
    <default class="collision">
      <geom group="3" rgba="0.3 0.5 0.9 0.3" friction="0.9 0.005 0.0001"/>
    </default>
  </default>
 
  <worldbody>
    <light name="sun" pos="0 0 6" dir="-0.3 -0.3 -1" directional="true"
           diffuse="0.85 0.85 0.8" specular="0.3 0.3 0.3" castshadow="true"/>
    <light name="fill" pos="0 0 4" directional="false" diffuse="0.25 0.25 0.28"/>
 
    <geom name="floor" type="plane" size="25 25 0.1" material="floor_mat"/>
 
{bodies}
  </worldbody>
 
  <actuator>
{actuators}
  </actuator>
 
  <sensor>
{sensors}
  </sensor>
 
</mujoco>
"""
    with open(out_path, "w") as f:
        f.write(xml)
    print(f"Wrote {out_path} with {n} robots ({cols}x{math.ceil(n/cols)} grid, spacing={spacing}m)")
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16, help="number of robots (recommended 10-20 on this CPU)")
    p.add_argument("--spacing", type=float, default=2.0, help="meters between robots in the grid")
    p.add_argument("--out", type=str, default="piple_arena.xml")
    args = p.parse_args()
    build(args.n, args.spacing, args.out)
 
