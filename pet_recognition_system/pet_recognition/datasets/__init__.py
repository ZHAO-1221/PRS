"""Dataset loaders and data transforms."""

__all__ = ["DOG_BEHAVIOR_CLASSES", "DogBehaviorSkeletonDataset",
           "INTERACTION_CLASSES", "InteractionSkeletonDataset"]


def __getattr__(name):
    # Pose-only tools do not require PyTorch.
    if name in __all__:
        if name in {"INTERACTION_CLASSES", "InteractionSkeletonDataset"}:
            from .interaction import INTERACTION_CLASSES, InteractionSkeletonDataset
            return {"INTERACTION_CLASSES": INTERACTION_CLASSES,
                    "InteractionSkeletonDataset": InteractionSkeletonDataset}[name]
        from .dog_behavior import DOG_BEHAVIOR_CLASSES, DogBehaviorSkeletonDataset
        return {"DOG_BEHAVIOR_CLASSES": DOG_BEHAVIOR_CLASSES,
                "DogBehaviorSkeletonDataset": DogBehaviorSkeletonDataset}[name]
    raise AttributeError(name)

