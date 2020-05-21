class Printr:
    def __init__(self, width):
        self.width = width

    def printr(self, string):
        diff = self.width - len(string)
        print(end = "\r")
        if diff >= 0:
            print(end = string)
        else:
            print(end = string[:self.width - 3] + "...")
        print(end = " " * diff, flush = True)
