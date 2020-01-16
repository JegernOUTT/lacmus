from dataclasses import dataclass


__all__ = ['Size2D', 'Size3D', 'Size2DF', 'Size3DF']


@dataclass(frozen=True, order=True, eq=True, repr=True)
class Size2D:
    width: int
    height: int


@dataclass(frozen=True, order=True, eq=True, repr=True)
class Size3D:
    width: int
    height: int
    channels: int


@dataclass(frozen=True, order=True, eq=True, repr=True)
class Size2DF:
    width: float
    height: float


@dataclass(frozen=True, order=True, eq=True, repr=True)
class Size3DF:
    width: float
    height: float
    channels: float
