from typing import Optional, List


class Quality:
    def __init__(self, quality:str):
        self.quality:str = quality

    def getDegrees(self) -> Optional[List[int]]:
        if (self.quality == ""):           return [4, 7]
        if (self.quality == "m"):          return [3, 7]
        if (self.quality == "min"):        return [3, 7]
        if (self.quality == "minor"):      return [3, 7]
        if (self.quality == "dim"):        return [3, 6]
        if (self.quality == "diminished"): return [3, 6]
        if (self.quality == "aug"):        return [4, 8]
        if (self.quality == "augmented"):  return [4, 8]

        return None
