class OBJ:
    # location and scale properties
    # self.scale
    # self.offset
    #
    # # obj properties
    # self.vertecies
    # self.normals
    # self.faces

    def __init__(self, file_path, scale, offset):
        self.scale = scale
        self.offset = offset

        self.vertecies = []
        self.normals = []
        self.faces = []
        for line in open(file_path):
            if line.startswith('#'):
                continue

            line = line.split()
            if not line or len(line) < 1:
                continue

            if line[0] == "v":
                self.vertecies.append(map(float, line[1:4]))
            elif line[0] == "vn":
                self.normals.append(map(float, line[1:4]))
            elif line[0] == "f":
                face = []
                for vert in line[1:]:
                    vert = vert.split('/')
                    # vertex, texture(ignored for now), normal
                    face.append((vert[0], None, vert[2]))

                self.faces.append(face)
