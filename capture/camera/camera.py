# Imports
from abc import ABC, abstractmethod

class Camera(ABC):
    @abstractmethod
    def capture_frame(self):
        pass

    @abstractmethod
    def close(self):
        pass