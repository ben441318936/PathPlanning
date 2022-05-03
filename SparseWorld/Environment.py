import numpy as np

from MotionModels import MotionModel, DifferentialDrive

class Environment():
    def __init__(self, env_size=(100,100), motion_model: MotionModel = None) -> None:
        self.env_size = env_size
        self.motion_model = motion_model
        self.agent_state = np.zeros((self.motion_model.state_dim))
        self.set_agent_position(np.array([env_size[0]/2, env_size[1]/2]))

    def set_agent_state(self, state: np.ndarray) -> None:
        self.agent_state = state
    
    def set_agent_position(self, position: np.ndarray) -> None:
        self.agent_state[0:2] = position

    def get_agent_position(self) -> np.ndarray:
        return self.motion_model.state_2_position(self.agent_state)

    def get_agent_heading(self) -> np.ndarray:
        return self.motion_model.state_2_heading(self.agent_state)

    def get_agent_yaw_rate(self) -> np.ndarray:
        return self.motion_model.state_2_yaw_rate(self.agent_state)

    def get_agent_velocity(self) -> np.ndarray:
        return self.motion_model.state_2_velocity(self.agent_state)

    def agent_step(self, input, braking=False) -> bool:
        new_state = self.motion_model.step(self.agent_state, input, braking)
        if not self.state_out_of_bounds(new_state) and not self.state_collision(new_state):
            self.agent_state = new_state
            return True
        else:
            return False

    def state_out_of_bounds(self, state) -> bool:
        return state[0] < 0 or state[0] > self.env_size[0] or state[1] < 0 or state[1] > self.env_size[1]

    def state_collision(self, state) -> bool:
        return False


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    M = DifferentialDrive(sampling_period=0.1)
    E = Environment(motion_model=M)

    print(E.get_agent_position())

    pos = []
    vel = []

    for i in range(50):
        pos.append(E.get_agent_position())
        vel.append(E.get_agent_velocity())
        E.agent_step(input=np.array([0.1,0.5]))

    pos = np.array(pos)
    vel = np.array(vel)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(pos[:,0], pos[:,1])
    plt.axis("equal")
    plt.subplot(1,2,2)
    plt.plot(vel)
    plt.show()