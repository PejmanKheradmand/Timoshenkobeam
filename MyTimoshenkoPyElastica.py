import numpy as np
from elastica import *
from tqdm import tqdm
# Import modules
from elastica.modules import BaseSystemCollection, Constraints, Forcing, Damping, CallBacks

# Import Cosserat Rod Class
from elastica.rod.cosserat_rod import CosseratRod

# Import Damping Class
from elastica.dissipation import AnalyticalLinearDamper

# Import Boundary Condition Classes
from elastica.boundary_conditions import OneEndFixedRod, FreeRod
from elastica.external_forces import EndpointForces

# Import Timestepping Functions
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

import matplotlib.pyplot as plt

class TimoshenkoBeamSimulator(
    BaseSystemCollection, Constraints, Forcing, Damping, CallBacks):
    pass


timoshenko_sim = TimoshenkoBeamSimulator()

final_time = 10.0
n_elem = 100
density = 6450
E = 50e9
poisson_ratio = 0.33
shear_modulus = E / (2.0*(poisson_ratio + 1.0))
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 0.25
base_radius = 0.0014
base_area = np.pi * base_radius ** 2
nu = 1
dl = base_length / n_elem
dt = 0.01 * dl
'''
n_elem = 100
density = 1000
E = 1e6
poisson_ratio = 99
shear_modulus = E / ((poisson_ratio + 1.0))
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 3
base_radius = 0.25
base_area = np.pi * base_radius ** 2
nu = 1e-4
dl = base_length / n_elem
dt = 0.01 * dl
'''

shearable_rod = CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    0.0,     # internal damping constant, deprecated in v0.3.0
    E,
    shear_modulus=shear_modulus,
)

timoshenko_sim.append(shearable_rod)



timoshenko_sim.dampen(shearable_rod).using(
    AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)

timoshenko_sim.constrain(shearable_rod).using(
    OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
)
print("One end of the rod is now fixed in place")


origin_force = np.array([0.0, 0.0, 0.0])
end_force = np.array([0.10, 0.0, 0.0])


ramp_up_time = 1

timoshenko_sim.add_forcing_to(shearable_rod).using(
    EndpointForces, origin_force, end_force, ramp_up_time=ramp_up_time
)
print("Forces added to the rod")

rendering_fps = 60
step_skip = int(1.0 / (rendering_fps * dt))
class MyCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    # This function is called every time step
    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            # Save time, step number, position, orientation and velocity
            self.callback_params["time"].append(time)
            self.callback_params["position"].append(system.position_collection.copy())
            return


recorded_history = defaultdict(list)

timoshenko_sim.collect_diagnostics(shearable_rod).using(
    MyCallBack, step_skip=step_skip, callback_params=recorded_history)


timoshenko_sim.finalize()
print("System finalized")


total_steps = int(final_time / dt)
print("Total steps to take", total_steps)




timestepper = PositionVerlet()

integrate(timestepper, timoshenko_sim, final_time, total_steps)

Position = recorded_history["position"]
FinalPosition = Position[len(Position) - 1]
FinalPositionX = FinalPosition[0]
FinalPositionZ = FinalPosition[2]
plt.plot(FinalPositionX, FinalPositionZ)

plt.xlim([-1.6, 1.6])
plt.ylim([0, 3.2])
plt.show()



def plot_video(
    plot_params: dict,
    video_name="video.mp4",
    fps=15,
    xlim=(-0.5, 0.5),
    ylim=(0, 1),
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    positions_over_time = np.array(plot_params["position"])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x [m]", fontsize=16)
    ax.set_ylabel("z [m]", fontsize=16)
    rod_lines_2d = ax.plot(positions_over_time[0][0], positions_over_time[0][2])[0]
    # plt.axis("equal")
    with writer.saving(fig, video_name, dpi=150):
        for time in tqdm(range(1, len(plot_params["time"]))):
            rod_lines_2d.set_xdata(positions_over_time[time][0])
            rod_lines_2d.set_ydata(positions_over_time[time][2])
            writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())



filename_video = "Timoshenko.mp4"
plot_video(
    recorded_history,
    video_name=filename_video,
    fps=rendering_fps,
    xlim=(-1.6, 1.6),
    ylim=(0, 3.2),
)
