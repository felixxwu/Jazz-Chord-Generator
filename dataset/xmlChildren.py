def getChildren(element, tag):
    children = [c for c in element if c.tag == tag]
    return children

def getChild(element, tag):
    if (len(getChildren(element, tag)) == 0): return None
    return getChildren(element, tag)[0]