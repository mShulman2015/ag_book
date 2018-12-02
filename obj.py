import os
import numpy as np

class OBJ:
    # location and scale properties
    # self.scale
    # self.offset
    #
    # # obj properties
    # self.vertices
    # self.normals
    # self.faces

    def __init__(self, file_path, scale, offset):
        self.scale = scale
        self.offset = offset

        if not os.path.isfile(file_path):
            print("Could not load output obj: {}".format(file_path))
            return

        self.vertices = []
        self.normals = []
        self.faces = []
        for line in open(file_path):
            if line.startswith('#'):
                continue
            line = line.split()
            if not line or len(line) < 1:
                continue

            if line[0] == "v":
                self.vertices.append(np.array( [float(line[1]), float(line[2]), float(line[3])] ))
            elif line[0] == "vn":
                self.normals.append(np.array( [float(line[1]), float(line[2]), float(line[3])] ))
            elif line[0] == "f":
                face = []
                texcoords = []
                norms = []
                for v in line[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)

                self.faces.append((face, norms, texcoords))
        print("Loaded obj file: {}  - {} vertices, {} faces".format(file_path, len(self.vertices), len(self.faces)))
