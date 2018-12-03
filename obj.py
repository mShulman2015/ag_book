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

    def __init__(self, file_dir, info, input_dimentions):
        self.scale = np.array(list(map(float, info["scale"])))
        self.offset = np.array(list(map(float, info["offset"])))
        colors = list(map(int, list(info["display_color"])))
        self.display_color = (colors[0], colors[1], colors[2])

        re_orderer = list(map(int, info["axies_index"]))

        file_path = file_dir + info["file_name"]
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
                self.vertices.append(np.array([float(line[1]), float(line[2]), float(line[3])])[re_orderer])
            elif line[0] == "vn":
                self.normals.append(np.array([float(line[1]), float(line[2]), float(line[3])])[re_orderer])
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
        self.vertices = np.multiply(np.array(self.vertices), self.scale) + (page_center + self.offset)
        self.normals = np.array(self.normals)
        print("Loaded obj file: {}  - {} vertices, {} faces".format(file_path, len(self.vertices), len(self.faces)))
