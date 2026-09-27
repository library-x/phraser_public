from dataclasses import dataclass


@dataclass
class Segment:
  start: float
  end: float
  label: str
