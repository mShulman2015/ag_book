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

    def __init__(self, file_path, scale, offset, input_dimentions, display_color, swap_y_z=True):
        self.scale = scale
        self.offset = np.array(offset, dtype=np.float64)
        self.display_color = display_color

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
                if not swap_y_z:
                    self.vertices.append([float(line[1]), float(line[2]), float(line[3])])
                else:
                    self.vertices.append([float(line[1]), float(line[3]), float(line[2])])
            elif line[0] == "vn":
                if not swap_y_z:
                    self.normals.append([float(line[1]), float(line[2]), float(line[3])])
                else:
                    self.normals.append([float(line[1]), float(line[3]), float(line[2])])
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
        page_center = np.array([input_dimentions[1]/2, input_dimentions[0]/2, 0])
        self.vertices = np.array(self.vertices) * self.scale + (page_center + offset)
        self.normals = np.array(self.normals)
        print("Loaded obj file: {}  - {} vertices, {} faces".format(file_path, len(self.vertices), len(self.faces)))
