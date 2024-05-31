class GriddlyVariable():
    def __init__(self, name):
        self.name = name

    def val(self):
        return self.name

    def add(self, value):
        return {"add": [self.name, value]}

    def sub(self, value):
        return {"sub": [self.name, value]}

    def set(self, value):
        return {"set": [self.name, value]}

    def incr(self):
        return {"incr": self.name}

    def decr(self):
        return {"decr": self.name}

    def eq(self, value):
        return {"eq": [self.name, value]}

    def neq(self, value):
        return {"neq": [self.name, value]}

    def gt(self, value):
        return {"gt": [self.name, value]}

    def gte(self, value):
        return {"gte": [self.name, value]}

    def lt(self, value):
        return {"lt": [self.name, value]}

    def lte(self, value):
        return {"lte": [self.name, value]}
