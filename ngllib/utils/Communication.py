from ..environment import Environment
from abc import ABC, abstractmethod
import os
import msgpack
import pickle

class CommunicationProtocol(ABC):
    """
    Class for defining a communication protocol between a NGLClient and NGLServer.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the communication medium with necessary parameters (file paths, socket ports, etc.)
        """

    @abstractmethod
    def write_actions(self):
        """
        Write the actions to the communication medium.
        """
        pass

    @abstractmethod
    def write_observations(self):
        """
        Write the observations to the communication medium.
        """
        pass

    @abstractmethod
    def read_actions(self):
        """
        Read the actions from the communication medium.
        """
        pass

    @abstractmethod
    def read_observations(self):
        """
        Read the observations from the communication medium.
        """
        pass

class FilesystemProtocol(CommunicationProtocol):
    """
    Communication protocol that uses the filesystem to exchange information between the NGLClient and NGLServer.
    """

    def __init__(self, action_file_path:str, observation_file_path:str, timeout=50):
        self.action_path = action_file_path
        os.makedirs(self.action_path, exist_ok=True)
        self.observation_path = observation_file_path
        os.makedirs(self.observation_path, exist_ok=True)
        
        self.timeout = timeout

    def clear_actions(self, id):
        """
        Reset the action files for a given instance ID.
        """
        # Reset any files in the read state
        src = os.path.join(self.action_path, f"{id}_0")
        dst = os.path.join(self.action_path, f"{id}_1")
        if os.path.exists(src):
            os.replace(src, dst)

    def clear_observations(self, id):
        """
        Reset the observation files for a given instance ID.
        """
        # Reset any files in the read state
        src = os.path.join(self.observation_path, f"{id}_0")
        dst = os.path.join(self.observation_path, f"{id}_1")
        if os.path.exists(src):
            os.replace(src, dst)

    def write_actions(self, actions, id):
        """
        Write the actions to the action file.
        """        
        action_file = os.path.join(self.action_path, f"{id}_1")

        with open(action_file, "wb") as file:
            file.write(msgpack.packb(actions, use_bin_type=True))
        os.rename(action_file, os.path.join(self.action_path, f"{id}_0"))


    def read_actions(self,id):
        """
        Check for new actions from the action file.
        """
        tries = 0
        action_file = os.path.join(self.action_path, f"{id}_0")

        while tries < self.timeout:
            try:
                with open(action_file, "rb") as file:
                    actions = msgpack.unpackb(file.read(), raw=False)
                os.rename(action_file, os.path.join(self.action_path, f"{id}_1"))

                return actions
            except:
                tries += 1
        raise TimeoutError(f"No actions received for ID {id} after {self.timeout} attempts.")

    def write_observations(self, observations, id):
        """
        Write the observations to the observation file.
        """
        observation_file = os.path.join(self.observation_path, f"{id}_1")

        with open(observation_file, "wb") as file:
            pickle.dump(observations, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename(observation_file, os.path.join(self.observation_path, f"{id}_0"))

    def read_observations(self, id):
        """
        Check for new observations from the observation file.
        """
        tries = 0
        observation_file = os.path.join(self.observation_path, f"{id}_0")

        while tries < self.timeout:
            try:
                with open(observation_file, "rb") as file:
                    observations = pickle.load(file)
                os.rename(observation_file, os.path.join(self.observation_path, f"{id}_1"))
                return observations
            except:
                tries += 1
        raise TimeoutError(f"No observations received for ID {id} after {self.timeout} attempts.")

class NGLClient:
    """
    Class for sending actions and requesting observations from Neuroglancer across different processes.
    """

    _id = 1

    def __init__(self, protocol: CommunicationProtocol):
        self.protocol = protocol

        # Assign a UUID
        self.id = NGLClient._id
        NGLClient._id += 1

        self.protocol.clear_observations(self.id)

    def send_actions(self,actions:list):
        self.protocol.write_actions(actions,self.id)
        
        return self.protocol.read_observations(self.id)

class NGLServer:
    """
    Class for receiving actions and sending observations from Neuroglancer across different processes.
    """

    _id = 1

    def __init__(self, protocol: CommunicationProtocol, environment: Environment):
        self.protocol = protocol
        self.environment = environment

        # Assign a UUID
        self.id = NGLServer._id
        NGLServer._id += 1

        self.protocol.clear_actions(self.id)

    def process_actions(self):
        actions = self.protocol.read_actions(self.id)
        return self.protocol.write_observations(self.environment.step(actions), self.id)

    def start_session(self, **options:dict):
        self.environment.start_session(**options)







# TO-DO

class NGLServerManger:
    """
    Class for managing and coordinating multiple NGLServer instances.
    """

    def __init__(self):
        pass

class NGLClientManager:
    """
    Class for managing and coordinating multiple NGLClient instances.
    """

    def __init__(self):
        pass