import math
import numpy as np
import random
import cv2
import time

class universe:
    def __init__(self, n_elements):
        self.num = n_elements
        self.elts = np.empty(shape=(n_elements, 3), dtype=int)
        for i in range(n_elements):
            self.elts[i, 0] = 0
            self.elts[i, 1] = 1
            self.elts[i, 2] = i

    def size(self, x):
        return self.elts[x, 1]

    def num_sets(self):
        return self.num

    def find(self, x):
        y = int(x)
        while y != self.elts[y, 2]:
            y = self.elts[y, 2]
        self.elts[x, 2] = y
        return y

    def join(self, x, y):
        if self.elts[x, 0] > self.elts[y, 0]:
            self.elts[y, 2] = x
            self.elts[x, 1] += self.elts[y, 1]

        else:
            self.elts[x, 2] = y
            self.elts[y, 1] += self.elts[x, 1]
            if self.elts[x, 0] == self.elts[y, 0]:
                self.elts[y, 0] += 1

        self.num -= 1

def segment_graph(num_vertices, num_edges, edges, c):

    edges[0:num_edges, :] = edges[edges[0:num_edges, 2].argsort()]
    u = universe(num_vertices)

    threshold = np.zeros(shape=num_vertices, dtype=float)

    for i in range(num_vertices):
        threshold[i] = get_threshold(1, c)

    for i in range(num_edges):
        pedge = edges[i, :]

        a = u.find(pedge[0])
        b = u.find(pedge[1])
        if a != b:
            if (pedge[2] < threshold[a]) and (pedge[2] <= threshold[b]):
                u.join(a, b)
                a = u.find(a)
                threshold[a] = pedge[2] + get_threshold(u.size(a), c)

    return u

def get_threshold(size, c):
    return c / size

def diff(r, g, b, x1, y1, x2, y2):
    res = math.sqrt((r[y1, x1] - r[y2, x2])**2 + (g[y1, x1] - g[y2, x2])**2 + (b[y1, x1] - b[y2, x2])**2)
    return res

def segment(in_image, k, min_size):

    h, w, c = in_image.shape
    
    edges_size = w * h * 4
    edges = np.zeros(shape=(edges_size, 3), dtype=object)
    num = 0

    b = in_image[:, :, 0].astype(np.float32)
    g = in_image[:, :, 1].astype(np.float32)
    r = in_image[:, :, 2].astype(np.float32)

    for y in range(h):
        for x in range(w):
            if x < w - 1:
                edges[num, 0] = int(y * w + x)
                edges[num, 1] = int(y * w + x + 1)
                edges[num, 2] = diff(r, g, b, x, y, x+1, y)
                num += 1
            if y < h - 1:
                edges[num, 0] = int(y * w + x)
                edges[num, 1] = int((y+1) * w + x )
                edges[num, 2] = diff(r, g, b, x, y, x, y+1)
                num += 1
            if x < w - 1 and y < h - 2:
                edges[num, 0] = int(y * w + x)
                edges[num, 1] = int((y+1) * w + x + 1)
                edges[num, 2] = diff(r, g, b, x, y, x+1, y+1)
                num += 1
            if x < w - 1 and y > 0:
                edges[num, 0] = int(y * w + x)
                edges[num, 1] = int((y-1) * w + x + 1)
                edges[num, 2] = diff(r, g, b, x, y, x+1, y-1)
                num += 1


    u = segment_graph(w*h, num, edges, k)

    for i in range(num):
        a = u.find(edges[i, 0])
        b = u.find(edges[i, 1])
        if (a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)):
            u.join(a, b)

    num_cc = u.num_sets()
    output = np.zeros(shape=(h, w, 3))

    colors = np.random.randint(0, 255, size=(h*w, 3))

    for y in range(h):
        for x in range(w):
            comp = u.find(y*w+x)
            output[y, x, :] = colors[comp, :]

    return output

if __name__ == '__main__':
    img = cv2.imread('../../data/VOC2007/JPEGImages/000001.jpg')
    img2 = cv2.GaussianBlur(img, (3, 3), 0)
    img2 = img2.astype(np.float32) / 255. 
    ss = time.time()
    output = segment(img, 40., 50)
    ee = time.time()
    print(ee-ss)
    res = np.concatenate([img, output], axis=1)
    cv2.imwrite('aaa.jpg', res)

