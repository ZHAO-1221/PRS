"""Model definitions and wrappers."""

__all__ = ["DogBehaviorMotionAuxTCN"]


def __getattr__(name):
    if name == "DogBehaviorMotionAuxTCN":
        from .skeleton_tcn import DogBehaviorMotionAuxTCN
        return DogBehaviorMotionAuxTCN
    raise AttributeError(name)

