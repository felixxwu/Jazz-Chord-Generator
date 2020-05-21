from typing import Optional, List

class Extension:
    def __init__(self, extension:str):
        self.extension:str = extension

    def getDegrees(self) -> Optional[List[int]]:
        if self.extension == "6":       return [9]
        if self.extension == "69":      return [9, 14]
        if self.extension == "6/9":     return [9, 14]
        if self.extension == "6/11":    return [9, 17]
        if self.extension == "6/13":    return [9, 21]
        if self.extension == "7":       return [10]
        if self.extension == "9":       return [10, 14]
        if self.extension == "11":      return [10, 17]
        if self.extension == "13":      return [10, 21]
        if self.extension == "M7":      return [11]
        if self.extension == "maj7":    return [11]
        if self.extension == "major7":  return [11]
        if self.extension == "M9":      return [11, 14]
        if self.extension == "maj9":    return [11, 14]
        if self.extension == "major9":  return [11, 14]
        if self.extension == "M11":     return [11, 17]
        if self.extension == "maj11":   return [11, 17]
        if self.extension == "major11": return [11, 17]
        if self.extension == "M13":     return [11, 21]
        if self.extension == "maj13":   return [11, 21]
        if self.extension == "major13": return [11, 21]

        return None
