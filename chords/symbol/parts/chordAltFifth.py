from typing import Optional

class Fifth:
    def __init__(self, alteration:str):
        self.alteration:str = alteration

    def getAlterations(self) -> Optional[dict]:
        if self.alteration == "b5":     return {"+": [6], "-": [7]}
        if self.alteration == "(b5)":   return {"+": [6], "-": [7]}
        if self.alteration == "#5":     return {"+": [8], "-": [7]}
        if self.alteration == "(#5)":   return {"+": [8], "-": [7]}

        if self.alteration == "+b9":    return {"+": [1]}
        if self.alteration == "(+b9)":  return {"+": [1]}
        if self.alteration == "+9":     return {"+": [2]}
        if self.alteration == "(+9)":   return {"+": [2]}
        if self.alteration == "+#9":    return {"+": [3]}
        if self.alteration == "(+#9)":  return {"+": [3]}
        
        if self.alteration == "+11":    return {"+": [5]}
        if self.alteration == "(+11)":  return {"+": [5]}
        if self.alteration == "+#11":   return {"+": [6]}
        if self.alteration == "(+#11)": return {"+": [6]}
        
        if self.alteration == "+b13":   return {"+": [8]}
        if self.alteration == "(+b13)": return {"+": [8]}

        if self.alteration == "+7":     return {"+": [11]}
        if self.alteration == "(+7)":   return {"+": [11]}

        return None
