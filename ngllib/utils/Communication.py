from ..environment import Environment
from abc import ABC, abstractmethod
import os
import msgpack
import pickle
import socket
import struct
import threading
import time


class CommunicationProtocol(ABC):
    """
    Class for defining a communication protocol between a NGLClient and NGLServer.
    """

    @abstractmethod
    def __init__(self):
        """
        Initialize the communication medium.
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

    @abstractmethod
    def read_observations_silent(self):
        """
        Read the first observation without consuming it (filesystem) or
        equivalently just read it (socket).
        """
        pass


class FilesystemProtocol(CommunicationProtocol):
    """
    Communication protocol that uses the filesystem to exchange information between the NGLClient and NGLServer.

    Args:
    - action_file_path: The directory path where action files will be stored.
    - observation_file_path: The directory path where observation files will be stored.
    - timeout: The maximum number of attempts to read from the files.
    """

    def __init__(self, action_file_path: str, observation_file_path: str, timeout=50):
        self.action_path = action_file_path
        os.makedirs(self.action_path, exist_ok=True)
        self.observation_path = observation_file_path
        os.makedirs(self.observation_path, exist_ok=True)

        self.timeout = timeout

    def clear_actions(self, id):
        """
        Reset the action files for a given instance ID.
        """
        src = os.path.join(self.action_path, f"{id}_0")
        dst = os.path.join(self.action_path, f"{id}_1")
        if os.path.exists(src):
            os.replace(src, dst)

    def clear_observations(self, id):
        """
        Reset the observation files for a given instance ID.
        """
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


    def read_actions(self, id):
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

    def read_observations_silent(self, id):
        """
        Check for first observation from the observation file without renaming.
        """
        tries = 0
        observation_file = os.path.join(self.observation_path, f"{id}_0")

        while tries < self.timeout:
            try:
                with open(observation_file, "rb") as file:
                    observations = pickle.load(file)
                return observations
            except:
                tries += 1
        raise TimeoutError(f"No observations received for ID {id} after {self.timeout} attempts.")
        

# ---------------------------------------------------------------------------
# Socket-based IPC
# ---------------------------------------------------------------------------

def _send_message(sock: socket.socket, data: bytes):
    """Send a length-prefixed message over a socket.

    Protocol: 4-byte big-endian length header followed by the payload.
    This guarantees the receiver knows exactly how many bytes to read,
    which avoids the classic TCP framing problem where multiple small
    writes can arrive merged or split across recv() calls.
    """
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)


def _recv_message(sock: socket.socket) -> bytes:
    """Receive a length-prefixed message from a socket.

    Blocks until the full message arrives.  Returns the raw payload bytes.
    Raises ``ConnectionError`` if the peer closes the connection mid-message.
    """
    # Read the 4-byte length header
    header = bytearray(4)
    view = memoryview(header)
    received = 0
    while received < 4:
        n = sock.recv_into(view[received:])
        if not n:
            raise ConnectionError("Connection closed while reading message header")
        received += n

    (length,) = struct.unpack("!I", header)

    # Read exactly `length` bytes of payload into a pre-allocated buffer
    payload = bytearray(length)
    view = memoryview(payload)
    received = 0
    while received < length:
        n = sock.recv_into(view[received:])
        if not n:
            raise ConnectionError("Connection closed while reading message payload")
        received += n

    return bytes(payload)


class SocketProtocol(CommunicationProtocol):
    """Communication protocol that uses TCP sockets for IPC.

    One side acts as the **listener** (``is_server=True``) and the other as
    the **connector** (``is_server=False``).  In the typical NeuroGym setup
    the ``NGLServer`` (environment process) should be the listener and the
    ``NGLClient`` (agent / controller process) should be the connector so
    that the environment can be started first and wait for clients.

    Data is serialized with ``pickle`` (observations contain PIL images and
    numpy arrays that pickle handles well) and ``msgpack`` (actions are
    simple numeric lists).  Each message is length-prefixed so TCP
    framing is handled correctly.

    Args:
        host: The hostname / IP to bind or connect to.
        port: The TCP port number.
        is_server: If True, bind and listen for connections (environment side).
                   If False, connect to a listening server (agent side).
        timeout: Seconds to wait when connecting or waiting for a peer
                 before raising ``TimeoutError``.
    """

    def __init__(self, host: str = "localhost", port: int = 5555,
                 is_server: bool = True, timeout: float = 60.0):
        self.host = host
        self.port = port
        self.is_server = is_server
        self.timeout = timeout

        # Will hold the connected socket for each client id.
        # For a single-client setup this maps {id: socket}.
        self._connections: dict[int, socket.socket] = {}

        # Server-side listener
        self._server_socket: socket.socket | None = None

        if is_server:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.bind((self.host, self.port))
            self._server_socket.listen(8)
            self._server_socket.settimeout(self.timeout)
            print(f"[SocketProtocol] Listening on {self.host}:{self.port}")

    # -- Connection management ------------------------------------------------

    def _accept_connection(self, id: int):
        """Block until a client connects and register it under *id*."""
        if id in self._connections:
            return  # already connected
        try:
            conn, addr = self._server_socket.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._connections[id] = conn
            print(f"[SocketProtocol] Accepted connection from {addr} for id={id}")
        except socket.timeout:
            raise TimeoutError(
                f"No client connected for id={id} within {self.timeout}s")

    def _connect(self, id: int):
        """Connect to the server and register the socket under *id*."""
        if id in self._connections:
            return  # already connected
        deadline = time.time() + self.timeout
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.connect((self.host, self.port))
                self._connections[id] = sock
                print(f"[SocketProtocol] Connected to {self.host}:{self.port} for id={id}")
                return
            except ConnectionRefusedError:
                sock.close()
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Could not connect to {self.host}:{self.port} "
                        f"for id={id} within {self.timeout}s")
                time.sleep(0.05)

    def _get_conn(self, id: int) -> socket.socket:
        """Return the socket for *id*, establishing the connection if needed."""
        if id not in self._connections:
            if self.is_server:
                self._accept_connection(id)
            else:
                self._connect(id)
        return self._connections[id]

    # -- CommunicationProtocol interface --------------------------------------

    def clear_actions(self, id):
        """No-op for sockets — there are no stale files to clean up."""
        pass

    def clear_observations(self, id):
        """No-op for sockets — there are no stale files to clean up."""
        pass

    def write_actions(self, actions, id):
        """Serialize *actions* with msgpack and send over the socket."""
        conn = self._get_conn(id)
        data = msgpack.packb(actions, use_bin_type=True)
        _send_message(conn, data)

    def read_actions(self, id):
        """Block until an action message arrives, deserialize and return it."""
        conn = self._get_conn(id)
        data = _recv_message(conn)
        actions = msgpack.unpackb(data, raw=False)
        return actions

    def write_observations(self, observations, id):
        """Serialize *observations* with pickle and send over the socket."""
        conn = self._get_conn(id)
        data = pickle.dumps(observations, protocol=pickle.HIGHEST_PROTOCOL)
        _send_message(conn, data)

    def read_observations(self, id):
        """Block until an observation message arrives, deserialize and return it."""
        conn = self._get_conn(id)
        data = _recv_message(conn)
        observations = pickle.loads(data)
        return observations

    def read_observations_silent(self, id):
        """Read observation — identical to read_observations for sockets.

        Sockets have no file to leave in place, so the \"silent\" distinction
        that the filesystem protocol needs does not apply here.
        """
        return self.read_observations(id)

    # -- Cleanup --------------------------------------------------------------

    def close(self):
        """Close all connections and the server socket."""
        for conn in self._connections.values():
            try:
                conn.close()
            except OSError:
                pass
        self._connections.clear()
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None


class NGLClient:
    """
    Class for sending actions and requesting observations from Neuroglancer across different processes.

    Args:
    - protocol: An instance of a CommunicationProtocol subclass.
    """

    _id = 1

    def __init__(self, protocol: CommunicationProtocol):
        self.protocol = protocol

        # Assign a UUID
        self.id = NGLClient._id
        NGLClient._id += 1

        self.protocol.clear_observations(self.id)

    def send_actions(self, actions: list):
        """Send an action to the server and block until the next observation arrives."""
        self.protocol.write_actions(actions, self.id)

        return self.protocol.read_observations(self.id)

    def get_initial(self):
        return self.protocol.read_observations_silent(self.id)

class NGLServer:
    """
    Class for receiving actions and sending observations from Neuroglancer across different processes.

    Args:
        protocol: An instance of a CommunicationProtocol subclass.
        environment: An Environment instance.
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

    def start_session(self, start_url=None, **options:dict):
        # Accept the client connection early so it doesn't time out
        # while we do the slow environment startup + screenshot.
        self.protocol._get_conn(self.id)

        self.environment.start_session(start_url=start_url,**options)

        # Get initial state and send to client
        state, json_state = self.environment.prepare_state()
        initial = [state, 0, False, json_state]

        self.protocol.write_observations(initial, self.id)



