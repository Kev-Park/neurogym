"""
Filesystem-based IPC with atomic writes to prevent read/write conflicts.

Usage:
    channel = IPCChannel("ipc/agent_0")

    # Writer side (e.g., environment process)
    channel.write_state(pos_state, image)

    # Reader side (e.g., controller process)
    pos_state, image = channel.read_state()
"""

import os
import io
import pickle
import tempfile
import fcntl
import numpy as np
from PIL import Image


def atomic_write(filepath, data_bytes, sync=False):
    """Write data to a file atomically using tmp file + rename.
    Args:
        sync: If True, fsync before rename (slower but durable across power loss).
              For IPC on the same machine, False is fine.
    """
    dirpath = os.path.dirname(filepath)
    fd, tmp_path = tempfile.mkstemp(dir=dirpath)
    try:
        os.write(fd, data_bytes)
        if sync:
            os.fsync(fd)
        os.close(fd)
        os.rename(tmp_path, filepath)  # atomic on same filesystem
    except:
        os.close(fd)
        os.unlink(tmp_path)
        raise


def locked_read(filepath):
    """Read a file with a shared lock (blocks if a writer holds exclusive lock)."""
    with open(filepath, 'rb') as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        data = f.read()
        fcntl.flock(f, fcntl.LOCK_UN)
    return data


class IPCChannel:
    def __init__(self, directory):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, name):
        return os.path.join(self.directory, name)

    # --- State (pos_state as pkl + image as npy) ---

    def write_state(self, pos_state, image):
        """Write state and image. Called by the environment process."""
        # pos_state as pickle
        atomic_write(self._path("state.pkl"), pickle.dumps(pos_state))
        # image as npy (convert PIL Image to numpy array)
        img_array = np.array(image)
        bio = io.BytesIO()
        np.save(bio, img_array)
        atomic_write(self._path("image.npy"), bio.getvalue())

    def read_state(self):
        """Read state and image. Called by the controller process.
        Returns (pos_state, PIL.Image) or (None, None) if not yet available."""
        state_path = self._path("state.pkl")
        image_path = self._path("image.npy")
        if not os.path.exists(state_path) or not os.path.exists(image_path):
            return None, None
        pos_state = pickle.loads(locked_read(state_path))
        img_array = np.load(io.BytesIO(locked_read(image_path)))
        image = Image.fromarray(img_array)
        return pos_state, image

    # --- Action (as pkl) ---

    def write_action(self, action):
        """Write action vector. Called by the controller process."""
        atomic_write(self._path("action.pkl"), pickle.dumps(action))

    def read_action(self):
        """Read action vector. Called by the environment process.
        Returns action list or None if not yet available."""
        action_path = self._path("action.pkl")
        if not os.path.exists(action_path):
            return None
        return pickle.loads(locked_read(action_path))

    # --- Ready flags (simple signaling) ---

    def signal(self, name):
        """Create a flag file to signal the other process."""
        open(self._path(f"{name}.flag"), 'w').close()

    def check_signal(self, name):
        """Check if a flag exists, and remove it if so. Returns True/False."""
        flag_path = self._path(f"{name}.flag")
        if os.path.exists(flag_path):
            os.unlink(flag_path)
            return True
        return False

    def clear(self):
        """Remove all files in the channel directory."""
        for f in os.listdir(self.directory):
            os.unlink(self._path(f))
