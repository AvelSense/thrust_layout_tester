import re
import select
import sys
import time
from queue import Queue, Empty
from threading import Thread, Event
import signal
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def make_line_cylinder(origin, length, direction, color, radius=0.001):
    if abs(length) < 1e-6:
        return None

    R = Rotation.from_euler('ZYX', direction, degrees=True)
    p1 = np.asarray(origin, dtype=float)
    p2 = p1 + R.apply(np.array([length, 0.0, 0.0]))

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=abs(length)
    )
    cylinder.paint_uniform_color(color)

    line_dir = (p2 - p1) / abs(length)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, line_dir)
    angle = np.arccos(np.clip(np.dot(z, line_dir), -1.0, 1.0))

    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cylinder.rotate(rot, center=[0, 0, 0])

    cylinder.translate((p1 + p2) / 2)
    return cylinder


def make_origin_axes(length=2.0, radius=0.001):
    return [
        make_line_cylinder([0, 0, 0], length, [0, 0, 0], [1, 0, 0], radius),
        make_line_cylinder([0, 0, 0], length, [90, 0, 0], [0, 1, 0], radius),
        make_line_cylinder([0, 0, 0], length, [0, -90, 0], [0, 0, 1], radius),
    ]


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def compute_unit_force_vector(yaw, pitch, roll):
    R = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True)
    return R.apply(np.array([1.0, 0.0, 0.0]))


def compute_unit_torque_vector(x, y, z, force_vector):
    uf = force_vector
    return np.array([
        y * uf[2] - z * uf[1],
        z * uf[0] - x * uf[2],
        x * uf[1] - y * uf[0],
    ])


def build_layout(motors, directions):
    columns = []
    for motor, direction in zip(motors, directions):
        uf = compute_unit_force_vector(*direction)
        ut = compute_unit_torque_vector(*motor, uf)
        columns.append(np.hstack((uf, ut)))
    return np.column_stack(columns)


# ---------------------------------------------------------------------------
# Non-blocking stdin reader
# ---------------------------------------------------------------------------

class StdinReader(Thread):
    PROMPT = "\nDesired [Fx, Fy, Fz, Mx, My, Mz] > "

    def __init__(self, cmd_queue: Queue, stop_event: Event):
        super().__init__(daemon=True)
        self.cmd_queue = cmd_queue
        self.stop_event = stop_event

    def run(self):
        sys.stdout.write(self.PROMPT)
        sys.stdout.flush()
        while not self.stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            line = sys.stdin.readline()
            if not line:
                self.stop_event.set()
                break
            line = line.strip()
            if line.lower() in ("quit", "exit", "stop"):
                self.stop_event.set()
                break
            self.cmd_queue.put(line)
            if not self.stop_event.is_set():
                time.sleep(0.3) # Wait for calculation to end
                sys.stdout.write(self.PROMPT)
                sys.stdout.flush()


# ---------------------------------------------------------------------------
# Command parser / solution computer
# ---------------------------------------------------------------------------

def parse_command(command: str):
    command = command.strip().lstrip('[').rstrip(']')
    numbers = list(map(float, re.split(r"[;,| ]+", command)))
    if len(numbers) != 6:
        raise ValueError(f"Expected 6 numbers, got {len(numbers)}.")
    return numbers


def compute_solution(fx, fy, fz, mx, my, mz, layout,
                     layout_inv, motors, directions, colors,
                     geom_queue, previous_cylinders):
    """
    Compute motor efforts, queue removal of previous solution cylinders,
    and queue addition of new ones. Returns the new cylinder list.
    """
    wrench = np.array([fx, fy, fz, mx, my, mz])
    solution = (layout_inv @ wrench.reshape(-1, 1)).ravel()
    print("Found solution: " + ", ".join(f"M{i + 1} {float(v):.3f}" for i, v in enumerate(solution)))
    print("Actual [Fx, Fy, Fz, Mx, My, Mz]:", np.round(layout@solution, 2))

    # Remove previous solution's cylinders
    for cyl in previous_cylinders:
        geom_queue.put(("remove", cyl))

    # Build and queue new cylinders
    new_cylinders = []
    for i, effort in enumerate(solution):
        cyl = make_line_cylinder(motors[i], effort, directions[i], colors[i], radius=0.01)
        if cyl is not None:
            geom_queue.put(("add", cyl))
            new_cylinders.append(cyl)

    return new_cylinders


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    colors = [
        [1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1],
        [0.6, 0.1, 0.1], [0.1, 0.6, 0.1], [0.1, 0.1, 0.6],
    ]

    # Position from the center of gravity, in meters. Any number of motors.
    motors = [np.array([0., 0.3, 0.]), # M1
              np.array([0., -0.3, 0.]), # M2
              np.array([0.5, 0, 0.3]), # M3
              np.array([0.5, 0.3, 0]), # M4
              np.array([-0.5, 0, -0.3]), # M5
              np.array([-0.5, -0.3, 0])] # M6

    # Yaw, pitch, roll, in degrees. One direction per motor.
    directions = [np.array([0, 0, 0])] # M1
    directions += [np.array([0, 0, 0])] # M2
    directions += [np.array([0, 20, 0])] # M3
    directions += [np.array([-10, -10, 0])] # M4
    directions += [np.array([0, 20, 0])] # M5
    directions += [np.array([-10, -10, 0])] # M6


    # print(motors)
    # print(directions)

    layout = build_layout(motors, directions)
    layout_inv = np.linalg.pinv(layout)


    print(np.round(layout, 3), '\n\n')
    # print(np.round(layout_inv, 2), '\n\n')

    print("Rank of layout: ", np.linalg.matrix_rank(layout))

    geom_queue = Queue()
    cmd_queue = Queue()
    stop_event = Event()
    # Holds the cylinders from the last solution so they can be removed next time
    solution_cylinders = []

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = [0.2, 0.2, 0.2]

    for geom in make_origin_axes():
        vis.add_geometry(geom)

    for motor, direction, color in zip(motors, directions, colors):
        cyl = make_line_cylinder(motor, 1.0, direction, color, radius=0.002)
        if cyl:
            vis.add_geometry(cyl, reset_bounding_box=False)

    stdin_reader = StdinReader(cmd_queue, stop_event)
    stdin_reader.start()

    try:
        while not stop_event.is_set():
            # 1. Drain geometry queue (add/remove)
            while True:
                try:
                    action, geom = geom_queue.get_nowait()
                    if action == "add":
                        vis.add_geometry(geom, reset_bounding_box=False)
                    elif action == "remove":
                        vis.remove_geometry(geom, reset_bounding_box=False)
                except Empty:
                    break

            # 2. Process commands
            while True:
                try:
                    cmd = cmd_queue.get_nowait()
                    try:
                        numbers = parse_command(cmd)
                        solution_cylinders = compute_solution(
                            *numbers, layout, layout_inv, motors, directions, colors,
                            geom_queue, solution_cylinders
                        )
                    except ValueError as e:
                        print(f"Input is invalid: {e}")
                except Empty:
                    break

            # 3. Render
            if not vis.poll_events():
                stop_event.set()
                break
            vis.update_renderer()

    except KeyboardInterrupt:
        print("Program interrupted")
    finally:
        vis.destroy_window()
        stop_event.set()
        stdin_reader.join(timeout=1.0)

    print("Program ended")


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()




    # m1_pos = Rotation.from_euler('X', 0, degrees=True).apply(np.array([0.5, 0, 0.2]))
    # m2_pos = Rotation.from_euler('X', 120, degrees=True).apply(np.array([0.5, 0, 0.2]))
    # m3_pos = Rotation.from_euler('X', 240, degrees=True).apply(np.array([0.5, 0, 0.2]))
    # m4_pos = Rotation.from_euler('X', 0, degrees=True).apply(np.array([-0.5, 0, 0.2]))
    # m5_pos = Rotation.from_euler('X', 120, degrees=True).apply(np.array([-0.5, 0, 0.2]))
    # m6_pos = Rotation.from_euler('X', 240, degrees=True).apply(np.array([-0.5, 0, 0.2]))
    #
    # m1_dir = (Rotation.from_euler('X', 0, degrees=True) *
    #           Rotation.from_euler('ZYX', [0, 30, 0], degrees=True)).as_euler("ZYX", degrees=True)
    # m2_dir = (Rotation.from_euler('X', 120, degrees=True) *
    #           Rotation.from_euler('ZYX', [0, 30, 0], degrees=True)).as_euler("ZYX", degrees=True)
    # m3_dir = (Rotation.from_euler('X', 240, degrees=True) *
    #           Rotation.from_euler('ZYX', [0, 30, 0], degrees=True)).as_euler("ZYX", degrees=True)
    # m4_dir = (Rotation.from_euler('X', 0, degrees=True) *
    #           Rotation.from_euler('ZYX', [0, -30, 0], degrees=True)).as_euler("ZYX", degrees=True)
    # m5_dir = (Rotation.from_euler('X', 120, degrees=True) *
    #           Rotation.from_euler('ZYX', [0, -30, 0], degrees=True)).as_euler("ZYX", degrees=True)
    # m6_dir = (Rotation.from_euler('X', 240, degrees=True) *
    #           Rotation.from_euler('ZYX', [0, -30, 0], degrees=True)).as_euler("ZYX", degrees=True)


    # directions = [np.array([0., 30., 0.]),
    #               np.array([26.56505118, -14.47751219, 116.56505118]),
    #               np.array([-26.56505118, -14.47751219, -116.56505118]),
    #               np.array([0., -30., 0.]),
    #               np.array([-26.56505118, 14.47751219, 116.56505118]),
    #               np.array([26.56505118, 14.47751219, -116.56505118])]


    # motors = [np.array([0.5, 0., 0.2]),
    #           np.array([0.5, -0.3, -0.1]),
    #           np.array([0.5, 0.3, -0.1]),
    #           np.array([-0.5, 0., 0.2]),
    #           np.array([-0.5, -0.3, -0.1]),
    #           np.array([-0.5, 0.3, -0.1])]


    # intersection_front = np.array([-1, 0., .5])
    # directions += [Rotation.align_vectors(intersection_front-motor, np.array([1,0,0]))[0]
    #               .as_euler("ZYX", degrees=True)
    #               for motor in motors[4:]]