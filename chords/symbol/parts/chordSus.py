from typing import Optional


class Sus:
    def __init__(self, sus:str):
        self.sus:str = sus

    def getAlterations(self) -> Optional[dict]:
        if self.sus == "":       return None
        if self.sus == "sus":    return {"-": [3,4]}
        if self.sus == "sus2":   return {"+": [2], "-": [3,4]}
        if self.sus == "sus4":   return {"+": [5], "-": [3,4]}
        if self.sus == "sus24":  return {"+": [2,5], "-": [3,4]}
        if self.sus == "sus2/4": return {"+": [2,5], "-": [3,4]}

        return None
