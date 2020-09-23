class Classes:
    # Load classes
    def __init__(self, path):
        file =  open(path, 'r')
        class_names = file.read().split()
        file.close()

        i2c = {}
        for index, name in enumerate(class_names):
            i2c[index] = name

        self.class_names = i2c
        self.number_of_classes = len(class_names)

    def class_name(self, index):
        if index in self.class_names:
            return self.class_names[index]
        else:
            return None
