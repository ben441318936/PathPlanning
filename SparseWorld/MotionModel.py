'''
Implements motion models for the robot.

Motion models define how a robot's state transitions with some known input.
They do not store state information for the robot.
Outside classes should implement state storage. 
For example, an Environment class should store the state of a robot moving in it.
An Agent class should store the (estimated) state of the robot as it moves, but 
might also stored a vectorized set of possible states, depending on the estimation method.

An abstract MotionModel class defines the basic interface and attributes of a motion model.
'''

from abc import ABC, abstractmethod
import numpy as np

class MotionModel(ABC):
    '''
    Defines MotionModel interface
    '''

    __slots__ = ("_tau", "_parameters", "_state_dim", "_input_dim")

    def __init__(self, sampling_period=1, parameters_dict=None) -> None:
        self._parameters = {}
        self._tau = sampling_period
        self._state_dim = 0
        self._input_dim = 0
        if parameters_dict is not None:
            self._parameters.update(parameters_dict)

    @property
    def sampling_period(self):
        return self._tau

    @property
    def parameters(self):
        return self._parameters

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def input_names(self) -> tuple:
        # return the input signl names
        # use them when constructing the input dict
        # or to determine ordering
        return None

    def create_input_dict(self, *inputs) -> dict:
        pass

    @abstractmethod
    def step(self, state: np.ndarray, input_dict: dict) -> None:
        pass

    def set_position(self, state: np.ndarray, pos: np.ndarray) -> None:
        return None

    def state_2_position(self, state: np.ndarray) -> np.ndarray:
        return None

    def set_heading(self, state: np.ndarray, heading: float) -> None:
        return None

    def state_2_heading(self, state: np.ndarray) -> np.ndarray:
        return None

    def set_pose(self, state: np.ndarray, pose: np.ndarray) -> None:
        return None

    def state_2_pose(self, state: np.ndarray) -> np.ndarray:
        return None

    def set_wheel_velocity(self, state: np.ndarray, velocity: np.ndarray) -> None:
        return None

    def state_2_wheel_velocity(self, state: np.ndarray) -> np.ndarray:
        return None
    
    def set_yaw_rate(self, state: np.ndarray, w: float) -> None:
        return None

    def state_2_yaw_rate(self, state: np.ndarray) -> np.ndarray:
        return None

    def set_velocity(self, state: np.ndarray, v: float) -> None:
        return None

    def state_2_velocity(self, state: np.ndarray) -> np.ndarray:
        return None


class DifferentialDriveVelocityInput(MotionModel):
    '''
    State is defined as [x,y,theta]:
        x: x coordinate
        y: y coordinate
        theta: heading

    Input is defined as [v,w]:
        v: speed in the direction of heading
        w: yaw rate (rate of change of heading)

    Use create_velocities_dict(v,w) for input formatting.
    '''

    __slots__ = ()

    def __init__(self, sampling_period=1, parameters_dict=None) -> None:
        params = {
            "wheel radius": 0.5,
            "axel length": 1,
        }
        if parameters_dict is not None:
            params.update(parameters_dict)
        super().__init__(sampling_period, params)
        self._state_dim = 3
        self._input_dim = 2

    @property
    def input_names(self) -> tuple:
        return ("v", "w")

    def create_input_dict(self, *inputs) -> dict:
        # takes two, v and w
        return {name: inp for (name, inp) in zip(self.input_names, inputs)}

    def step(self, state: np.ndarray, input_dict: dict) -> np.ndarray:
        '''
        State is defined as [x,y,theta]

        Input is defined as [v,w]

        Supports vectorized operations for N states and inputs
        States [N,3]
        v [N,1]
        w [N,1]
        '''

        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]

        v = input_dict["v"]
        w = input_dict["w"]

        new_state = state + self._tau * np.vstack((v * np.cos(state[:,2]),
                                                   v * np.sin(state[:,2]),
                                                   w)).T
        if N > 1:
            return new_state
        else:
            return np.squeeze(new_state, axis=0)

    def set_position(self, state: np.ndarray, pos: np.ndarray) -> None:
        if len(state.shape) > 1:
            state[:,0:2] = pos
        else:
            state[0:2] = pos

    def state_2_position(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,0:2]
        else:
            return state[0,0:2]

    def set_heading(self, state: np.ndarray, heading: float) -> None:
        if len(state.shape) > 1:
            state[:,2] = heading
        else:
            state[2] = heading

    def state_2_heading(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,2]
        else:
            return state[0,2]

    def set_pose(self, state: np.ndarray, pose: np.ndarray) -> None:
        if len(state.shape) > 1:
            state[:,0:3] = pose
        else:
            state[0:3] = pose

    def state_2_pose(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,0:3]
        else:
            return state[0,0:3]


class DifferentialDriveTorqueToVelocity(MotionModel):
    '''
    State is defined as [v,w]:
        v: linear velocity
        w: angular velocity (yaw rate)

    Input is defined as [T_R,T_L]:
        T_R: torque on right wheel
        T_L: torque on left wheel

    Use create_torque_dict(T_R,T_L) for input formatting.

    Parameters are:
        wheel radius
        robot mass: includes body and both wheels
        wheel friction: [0,1): controls how much the wheel velocity decays due to friction
        axel length
    '''

    __slots__ = ()

    def __init__(self, sampling_period=1, parameters_dict=None) -> None:
        params = {
            "robot mass": 20,
            "wheel friction": 0.5,
            "wheel radius": 0.5,
            "axel length": 1,
        }
        if parameters_dict is not None:
            params.update(parameters_dict)
        super().__init__(sampling_period, params)
        if "inertia" not in self._parameters:
            self._parameters["inertia"] = self._parameters["robot mass"] / 2 * self._parameters["wheel radius"]**2
        self._state_dim = 2
        self._input_dim = 2

    @property
    def input_names(self) -> tuple:
        return ("T_R", "T_L")

    def create_input_dict(self, *inputs) -> dict:
        # takes two, T_R, T_l
        return {name: inp for (name, inp) in zip(self.input_names, inputs)}

    def step(self, state: np.ndarray, input_dict: dict) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]

        v_dot = (input_dict["T_R"] + input_dict["T_L"]) * self._parameters["wheel radius"] / 2 / self._parameters["inertia"]
        w_dot = (input_dict["T_R"] - input_dict["T_L"]) * self._parameters["wheel radius"] / self._parameters["inertia"] / self._parameters["axel length"]
        
        phi_decay = np.exp(-self._parameters["wheel friction"] * self.sampling_period)
        phis_decay = np.array([phi_decay,phi_decay]).reshape((-1,self._state_dim))

        new_phis = phis_decay * state + self._tau * np.vstack((v_dot,
                                                               w_dot)).T

        if N > 1:
            return new_phis
        else:
            return np.squeeze(new_phis, axis=0)

    def state_2_wheel_velocity(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            phi_R = (state[:,0] / (self._parameters["wheel radius"] / 2) + state[:,1] / (self._parameters["wheel radius"] / self._parameters["axel length"])) / 2
            phi_L = (state[:,0] / (self._parameters["wheel radius"] / 2) - state[:,1] / (self._parameters["wheel radius"] / self._parameters["axel length"])) / 2
            return np.vstack((phi_R, phi_L))
        else:
            phi_R = (state[0,0] / (self._parameters["wheel radius"] / 2) + state[0,1] / (self._parameters["wheel radius"] / self._parameters["axel length"])) / 2
            phi_L = (state[0,0] / (self._parameters["wheel radius"] / 2) - state[0,1] / (self._parameters["wheel radius"] / self._parameters["axel length"])) / 2
            return np.array([phi_R, phi_L])
    
    def set_velocity(self, state: np.ndarray, v: float) -> None:
        if len(state.shape) > 1:
            state[:,0] = v
        else:
            state[0,0] = v

    def state_2_velocity(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,0]
        else:
            return state[0,0]

    def set_yaw_rate(self, state: np.ndarray, w: float) -> None:
        if len(state.shape) > 1:
            state[:,1] = w
        else:
            state[0,1] = w

    def state_2_yaw_rate(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,1]
        else:
            return state[0,1]


class DifferentialDriveTorqueToWheelVelocity(MotionModel):
    '''
    State is defined as [phi_R,phi_L]:
        phi_R: angular velocity of right wheel
        phi_L: angular velocity of left wheel

    Input is defined as [T_R,T_L]:
        T_R: torque on right wheel
        T_L: torque on left wheel

    Use create_torque_dict(T_R,T_L) for input formatting.

    Parameters are:
        wheel radius
        robot mass: includes body and both wheels
        wheel friction: [0,1): controls how much the wheel velocity decays due to friction
        axel length
    '''

    __slots__ = ()

    def __init__(self, sampling_period=1, parameters_dict=None) -> None:
        params = {
            "robot mass": 20,
            "wheel friction": 0.5,
            "wheel radius": 0.5,
            "axel length": 1,
        }
        if parameters_dict is not None:
            params.update(parameters_dict)
        super().__init__(sampling_period, params)
        if "inertia" not in self._parameters:
            self._parameters["inertia"] = self._parameters["robot mass"] / 2 * self._parameters["wheel radius"]**2
        self._state_dim = 2
        self._input_dim = 2

    @property
    def input_names(self) -> tuple:
        return ("T_R", "T_L")

    def create_input_dict(self, *inputs) -> dict:
        # takes two, T_R, T_l
        return {name: inp for (name, inp) in zip(self.input_names, inputs)}

    def step(self, state: np.ndarray, input_dict: dict) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        
        a_R = input_dict["T_R"] / self._parameters["inertia"]
        a_L = input_dict["T_L"] / self._parameters["inertia"]
        
        phi_decay = np.exp(-self._parameters["wheel friction"] * self.sampling_period)
        phis_decay = np.array([phi_decay,phi_decay]).reshape((-1,self._state_dim))

        new_phis = phis_decay * state + self._tau * np.vstack((a_R,
                                                               a_L)).T

        if N > 1:
            return new_phis
        else:
            return np.squeeze(new_phis, axis=0)

    def set_wheel_velocity(self, state: np.ndarray, velocity: np.ndarray) -> None:
        if len(state.shape) > 1:
            state[:,0:2] = velocity
        else:
            state[0:2] = velocity

    def state_2_wheel_velocity(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,0:2]
        else:
            return state[0,0:2]
    
    def state_2_yaw_rate(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return (state[:,0] - state[:,1]) * self._parameters["wheel radius"] / self._parameters["axel length"]
        else:
            return (state[0,0] - state[0,2]) * self._parameters["wheel radius"] / self._parameters["axel length"]

    def state_2_velocity(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return (state[:,0] + state[:,1]) * self._parameters["wheel radius"] / 2
        else:
            return (state[0,0] + state[0,1]) * self._parameters["wheel radius"] / 2


class DifferentialDriveTorqueInput(MotionModel):
    '''
    State is defined as [x,y,theta,v,w]:
        x: x coordinate
        y: y coordinate
        theta: heading
        v: linear velocity
        w: angular velocity

    Input is defined as [T_R,T_L]:
        T_R: torque on right wheel
        T_L: torque on left wheel

    Use create_torque_dict(T_R,T_L) for input formatting.

    Parameters are:
        wheel radius
        robot mass: includes body and both wheels
        axel length
        wheel friction: [0,1): controls how much the wheel velocity decays due to friction
    '''

    __slots__ = ("_velocity_input_submodel", "_torque_to_velocity_submodel")

    def __init__(self, sampling_period=1, parameters_dict=None) -> None:
        params = {
            "robot mass": 20,
            "wheel friction": 0.5,
            "wheel radius": 0.5,
            "axel length": 1,
        }
        if parameters_dict is not None:
            params.update(parameters_dict)
        super().__init__(sampling_period, params)
        if "inertia" not in self._parameters:
            self._parameters["inertia"] = self._parameters["robot mass"] / 2 * self._parameters["wheel radius"]**2
        self._velocity_input_submodel = DifferentialDriveVelocityInput(sampling_period, self._parameters)
        self._torque_to_velocity_submodel = DifferentialDriveTorqueToVelocity(sampling_period, self._parameters)
        self._state_dim = 5
        self._input_dim = 2

    @property
    def velocity_input_submodel(self) -> DifferentialDriveVelocityInput:
        return self._velocity_input_submodel

    @property
    def torque_to_velocity_submodel(self) -> DifferentialDriveTorqueToVelocity:
        return self._torque_to_velocity_submodel

    @property
    def input_names(self) -> tuple:
        return self._torque_to_velocity_submodel.input_names
        
    def create_input_dict(self, *inputs) -> dict:
        return self._torque_to_velocity_submodel.create_input_dict(inputs)

    def step(self, state: np.ndarray, input_dict: dict) -> np.ndarray:
        '''
        State is defined as [x,y,theta,v,w]
        Input is defined as [T_R,T_L]

        Supports vectorized operations for N states and inputs
        States [N,5]
        Inputs [N,2]
        '''

        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]

        # v = (state[:,3] + state[:,4]) * self._parameters["wheel radius"] / 2
        # w = (state[:,3] - state[:,4]) * self._parameters["wheel radius"] / self._parameters["axel length"]

        vel_inputs = [state[:,3], state[:,4]]
        v_w = {name: inp for (name, inp) in zip(self._velocity_input_submodel.input_names, vel_inputs)}

        new_pos_head = self._velocity_input_submodel.step(state[:,0:3], v_w)
        new_pos_head = new_pos_head.reshape((-1,self._velocity_input_submodel.state_dim))

        new_phis = self._torque_to_velocity_submodel.step(state[:,3:5], input_dict)

        # new_phis = self.torque_to_phi_step(state[:,3:5], input_dict)
        new_phis = new_phis.reshape((-1, self._torque_to_velocity_submodel.state_dim))

        new_state = np.hstack((new_pos_head, new_phis))

        if N > 1:
            return new_state
        else:
            return np.squeeze(new_state, axis=0)

    def set_position(self, state: np.ndarray, pos: np.ndarray) -> None:
        if len(state.shape) > 1:
            state[:,0:2] = pos
        else:
            state[0:2] = pos

    def state_2_position(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,0:2]
        else:
            return state[0,0:2]

    def set_heading(self, state: np.ndarray, heading: float) -> None:
        if len(state.shape) > 1:
            state[:,2] = heading
        else:
            state[2] = heading

    def state_2_heading(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,2]
        else:
            return state[0,2]

    def set_pose(self, state: np.ndarray, pose: np.ndarray) -> None:
        if len(state.shape) > 1:
            state[:,0:3] = pose
        else:
            state[0:3] = pose

    def state_2_pose(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,0:3]
        else:
            return state[0,0:3]

    def state_2_wheel_velocity(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            phi_R = (state[:,3] / (self._parameters["wheel radius"] / 2) + state[:,4] / (self._parameters["wheel radius"] / self._parameters["axel length"])) / 2
            phi_L = (state[:,3] / (self._parameters["wheel radius"] / 2) - state[:,4] / (self._parameters["wheel radius"] / self._parameters["axel length"])) / 2
            return np.vstack((phi_R, phi_L))
        else:
            phi_R = (state[0,3] / (self._parameters["wheel radius"] / 2) + state[0,4] / (self._parameters["wheel radius"] / self._parameters["axel length"])) / 2
            phi_L = (state[0,3] / (self._parameters["wheel radius"] / 2) - state[0,4] / (self._parameters["wheel radius"] / self._parameters["axel length"])) / 2
            return np.array([phi_R, phi_L])

    def set_velocity(self, state: np.ndarray, v: float) -> None:
        if len(state.shape) > 1:
            state[:,3] = v
        else:
            state[0,3] = v
    
    def state_2_velocity(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,3]
        else:
            return state[0,3]

    def set_yaw_rate(self, state: np.ndarray, w: float) -> None:
        if len(state.shape) > 1:
            state[:,4] = w
        else:
            state[0,4] = w
    
    def state_2_yaw_rate(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,4]
        else:
            return state[0,4]


class DifferentialDriveTorqueInputWheelvelocityState(MotionModel):
    '''
    State is defined as [x,y,theta,phi_R,phi_L]:
        x: x coordinate
        y: y coordinate
        theta: heading
        phi_R: angular velocity of right wheel
        phi_L: angular velocity of left wheel

    Input is defined as [T_R,T_L]:
        T_R: torque on right wheel
        T_L: torque on left wheel

    Use create_torque_dict(T_R,T_L) for input formatting.

    Parameters are:
        wheel radius
        robot mass: includes body and both wheels
        axel length
        wheel friction: [0,1): controls how much the wheel velocity decays due to friction
    '''

    __slots__ = ("_velocity_input_submodel", "_torque_to_velocity_submodel")

    def __init__(self, sampling_period=1, parameters_dict=None) -> None:
        params = {
            "robot mass": 20,
            "wheel friction": 0.5,
            "wheel radius": 0.5,
            "axel length": 1,
        }
        if parameters_dict is not None:
            params.update(parameters_dict)
        super().__init__(sampling_period, params)
        if "inertia" not in self._parameters:
            self._parameters["inertia"] = self._parameters["robot mass"] / 2 * self._parameters["wheel radius"]**2
        self._velocity_input_submodel = DifferentialDriveVelocityInput(sampling_period, self._parameters)
        self._torque_to_velocity_submodel = DifferentialDriveTorqueToWheelVelocity(sampling_period, self._parameters)
        self._state_dim = 5
        self._input_dim = 2

    @property
    def velocity_input_submodel(self) -> DifferentialDriveVelocityInput:
        return self._velocity_input_submodel

    @property
    def torque_to_velocity_submodel(self) -> DifferentialDriveTorqueToWheelVelocity:
        return self._torque_to_velocity_submodel

    @property
    def input_names(self) -> tuple:
        return self._torque_to_velocity_submodel.input_names
        
    def create_input_dict(self, *inputs) -> dict:
        return self._torque_to_velocity_submodel.create_input_dict(inputs)

    def step(self, state: np.ndarray, input_dict: dict) -> np.ndarray:
        '''
        State is defined as [x,y,theta,phi_R,phi_L]
        Input is defined as [T_R,T_L]

        Supports vectorized operations for N states and inputs
        States [N,5]
        Inputs [N,2]
        '''

        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]

        v = (state[:,3] + state[:,4]) * self._parameters["wheel radius"] / 2
        w = (state[:,3] - state[:,4]) * self._parameters["wheel radius"] / self._parameters["axel length"]

        vel_inputs = [v,w]
        v_w = {name: inp for (name, inp) in zip(self._velocity_input_submodel.input_names, vel_inputs)}

        new_pos_head = self._velocity_input_submodel.step(state[:,0:3], v_w)
        new_pos_head = new_pos_head.reshape((-1,self._velocity_input_submodel.state_dim))

        new_phis = self._torque_to_velocity_submodel.step(state[:,3:5], input_dict)

        # new_phis = self.torque_to_phi_step(state[:,3:5], input_dict)
        new_phis = new_phis.reshape((-1, self._torque_to_velocity_submodel.state_dim))

        new_state = np.hstack((new_pos_head, new_phis))

        if N > 1:
            return new_state
        else:
            return np.squeeze(new_state, axis=0)

    def set_position(self, state: np.ndarray, pos: np.ndarray) -> None:
        if len(state.shape) > 1:
            state[:,0:2] = pos
        else:
            state[0:2] = pos

    def state_2_position(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,0:2]
        else:
            return state[0,0:2]

    def set_heading(self, state: np.ndarray, heading: float) -> None:
        if len(state.shape) > 1:
            state[:,2] = heading
        else:
            state[2] = heading

    def state_2_heading(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,2]
        else:
            return state[0,2]

    def set_pose(self, state: np.ndarray, pose: np.ndarray) -> None:
        if len(state.shape) > 1:
            state[:,0:3] = pose
        else:
            state[0:3] = pose

    def state_2_pose(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,0:3]
        else:
            return state[0,0:3]

    def state_2_wheel_velocity(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return state[:,3:5]
        else:
            return state[0,3:5]
    
    def state_2_yaw_rate(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return (state[:,3] - state[:,4]) * self._parameters["wheel radius"] / self._parameters["axel length"]
        else:
            return (state[0,3] - state[0,4]) * self._parameters["wheel radius"] / self._parameters["axel length"]

    def state_2_velocity(self, state: np.ndarray) -> np.ndarray:
        state = state.reshape((-1,self._state_dim))
        N = state.shape[0]
        if N > 1:
            return (state[:,3] + state[:,4]) * self._parameters["wheel radius"] / 2
        else:
            return (state[0,3] + state[0,4]) * self._parameters["wheel radius"] / 2
    
                                                        
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    M = DifferentialDriveTorqueInput(sampling_period=0.1)
    curr_state = np.array([[0,0,0,1,0.5],
                           [0,0,0,1,1]])
    # curr_state = np.array([0,0,0,1,0.5])

    pos = []
    vel = []

    for i in range(50):
        pos.append(M.state_2_position(curr_state))
        vel.append(M.state_2_velocity(curr_state))
        curr_state = M.step(curr_state, input_dict={"T_R": np.array([1,1]), "T_L": np.array([1,0.5])})
        # curr_state = M.step(curr_state, input_torque=np.array([0.5,0.5]))

    pos = np.array(pos)
    vel = np.array(vel)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(pos[:,0], pos[:,1])
    plt.axis("equal")
    plt.subplot(1,2,2)
    plt.plot(vel)
    plt.show()
