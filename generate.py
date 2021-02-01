# Usage: python generate.py <card file name> <number of output images>
# ex. python generate.py BlueEyes.png 10

from PIL import Image, ImageColor, ImageEnhance
import sys
import math as math
import random
import numpy as np

def getBG(size, masks):
    layer1 = Image.new("RGBA", size, 
                       color=(random.randrange(0, 256), random.randrange(0, 256), 
                              random.randrange(0, 256), 255))
    layer2 = Image.new("RGBA", size, 
                       color=(random.randrange(0, 256), random.randrange(0, 256), 
                              random.randrange(0, 256), 255))
    mask = masks[random.randrange(0, len(masks))]
    layer1.paste(layer2, mask=mask)
    layer2.close()
    return layer1

# points = ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) rectangle
#           topleft,  topright, botright, botleft
# angle range (-pi/2, pi/2) uses right hand rule with thumb in -y direction (up)
def findHorizDest(points, angle):
    p1, p2, p3, p4 = points
    width = p2[0] - p1[0]
    height = p4[1] - p1[1]
    if angle < 0 and angle > -math.pi / 2:
        h_margin_max = height * 0.125    # arbitrary perspective shortening
        h_margin = math.sin(-angle) * h_margin_max
        w_margin = (-math.cos(angle) + 1) * width
        return np.array(((p1[0]+w_margin, p1[1]+h_margin), p2, 
                         p3, (p4[0]+w_margin, p4[1] - h_margin)), 
                        dtype=np.float64)
    elif angle < math.pi / 2:
        h_margin_max = height * 0.125
        h_margin = math.sin(angle) * h_margin_max
        res_w = width * math.cos(angle)
        return np.array((p1, (res_w, p2[1]+h_margin), (res_w, p3[1]-h_margin), p4), 
                        dtype=np.float64)
    else:
        return points

# points = ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) rectangle
#           topleft,  topright, botright, botleft
# angle range (-pi/2, pi/2) uses right hand rule with thumb in +x direction (right)
def findVertDest(points, angle):
    p1, p2, p3, p4 = points
    width = p2[0] - p1[0]
    height = p4[1] - p1[1]
    if angle < 0 and angle > -math.pi / 2:
        w_margin_max = width * 0.150    # arbitrary perspective shortening
        w_margin = math.sin(-angle) * w_margin_max
        h_margin = (-math.cos(angle) + 1) * height
        return np.array(((p1[0]+w_margin, p1[1]+h_margin), 
                         (p2[0]-w_margin, p2[1]+h_margin), p3, p4), 
                        dtype=np.float64)
    elif angle < math.pi / 2:
        w_margin_max = width * 0.150
        w_margin = math.sin(angle) * w_margin_max
        res_h = height * math.cos(angle)
        return np.array((p1, p2, (p3[0]-w_margin, res_h), (p4[0]+w_margin, res_h)), 
                        dtype=np.float64)
    else:
        return points


# points = ((x1, y1), (x2, y2), (x3, y3), (x4, y4)) rectangle
#           topleft,  topright, botright, botleft
# dirAngle range (-pi, pi) is direction of axis clockwise from parallel to x-axis
# rotAngle range (-pi/2, pi/2) is the rotation around the dirAngle axis, right hand rule
def findSphereDest(points, dirAngle, rotAngle):
    p1, p2, p3, p4 = points
    width, height = (p3[0]-p1[0], p3[1]-p1[1])
    center = (width / 2, height / 2)
    maxDepth = math.sqrt(center[0]*center[0] + center[1]*center[1])
    visSlope = math.tan(dirAngle)
    if abs(visSlope) > 1000000:    # arbitrary for infinite
        if dirAngle > 0:
            return findHorizDest(points, rotAngle) 
        else:
            return findHorizDest(points, -rotAngle)
    if abs(visSlope) < 0.00001:    # arbitrary for 0 (floating point)
        if abs(dirAngle) > math.pi / 2:
            return findVertDest(points, -rotAngle)
        else:
            return findVertDest(points, rotAngle) 
    ma = -visSlope   # translate to flipped coord plane (slope of Axis -> ma)
    res = []
    depths = []
    for p in points:
        x = p[0]
        y = p[1]
        xint = (0.5*(height - ma*width) - x/ma - y) / -((1/ma)+ma)
        yint = ma*(xint - width/2) + height/2
        ydiff = yint - y
        xdiff = xint - x
        y = y + ydiff*(-math.cos(rotAngle) + 1)
        x = x + xdiff*(-math.cos(rotAngle) + 1)
        res.append([x, y])
        # Determine depth (more positive = farther away)
        d = 0
        pointMaxDepth = math.sqrt(xdiff*xdiff + ydiff*ydiff)
        if dirAngle > 0 and dirAngle < math.pi:
            if xdiff > 0:
                d = pointMaxDepth * -math.sin(rotAngle)
            else:
                d = pointMaxDepth * math.sin(rotAngle)
        elif dirAngle < 0 and dirAngle > -math.pi:
            if xdiff > 0:
                d = pointMaxDepth * math.sin(rotAngle)
            else:
                d = pointMaxDepth * -math.sin(rotAngle)
        depths.append(d)
    # Depth distortion
    baseDepth = min(depths)
    modDepths = [d - baseDepth for d in depths]
    fullMaxDepth = maxDepth * 2
    maxShortenRatio = 5.5 / 8          # arbitrary perspective shortening
    for i, point in enumerate(res):
        shorten = (modDepths[i] / fullMaxDepth) * maxShortenRatio
        point[0] = point[0] + shorten * (center[0] - point[0])
        point[1] = point[1] + shorten * (center[1] - point[1])
    return np.array(res, dtype=np.float64)
    # direction of point movement is toward axis, perpendicular

# Fixes points from having negative coordinates
# Returns ndarray of new possibly shifted points
def padPoints(points):
    minY = 1000000
    minX = 1000000
    for p in points:
        minX = min(minX, p[0])
        minY = min(minY, p[1])
    for p in points:
        p[0] -= minX   # minX will be nonpositive
        p[1] -= minY
    return points

# points: tuple of 4 nonnegative points denoting the corners of a quadrilateral image
# Returns integer tuple (width, height) denoting the minimum size required for the image
def getSize(points):
    maxX = 0
    maxY = 0
    for p in points:
        maxX = max(maxX, p[0])
        maxY = max(maxY, p[1])
    return (math.ceil(maxX), math.ceil(maxY))


def findCoeffs(resultPts, srcPts):
    A = np.zeros((8, 8), dtype=np.float64)
    i = 0
    for p1, p2 in zip(resultPts, srcPts):
        A[i] = (p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1])
        i += 1
        A[i] = (0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1])
        i += 1

    B = srcPts.reshape(8)
    coeffs = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), B)
    return coeffs.reshape(8)

def tiltImg(im, dirAngle, rotAngle):
    src = np.array(((0, 0), (im.width, 0), (im.width, im.height), (0, im.height)),
                   dtype=np.float64)
    dest = findSphereDest(src, dirAngle, rotAngle)
    dest = padPoints(dest)
    newSize = getSize(dest)
    coeffs = findCoeffs(dest, src)
    transformed = im.transform(newSize, Image.PERSPECTIVE, coeffs)
    return transformed

targetCard = "BlueEyes.png"
numImages = 5
if len(sys.argv) >= 3:
    targetCard = str(sys.argv[1])
    numImages = int(sys.argv[2])
formatPlaces = math.ceil(math.log(numImages, 10))

outputSize = (1000, 1000)
noiseMasks = []
for i in range(5):
    noiseMasks.append(Image.effect_noise(outputSize, 100))

with Image.open(r"BlueEyes.png") as im:
    for i in range(numImages):
        tilted = tiltImg(im, random.random() * 2 * math.pi - math.pi, 
                             random.random() * 2 * math.pi / 3 - math.pi / 3)

        rotated = tilted.rotate(random.randrange(0, 360), expand=True)
        #bg = Image.new("RGBA", (1000, 1000), color="#69ba69")
        bg = getBG(outputSize, noiseMasks)

        lighter = ImageEnhance.Brightness(rotated)
        lighted = lighter.enhance(random.random() * 1.2 + 0.4)  # 0.4 to 1.6

        transformed = lighted   # can change easily
        # This section deals with scaling as ratio (1 = no scale)
        heightMaxScale = bg.height / transformed.height
        widthMaxScale = bg.width / transformed.width
        scaleMax = heightMaxScale if heightMaxScale < widthMaxScale else widthMaxScale
        heightMinScale = 0.125 * bg.height / transformed.height
        widthMinScale = 0.125 * bg.width / transformed.width
        scaleMin = heightMinScale if heightMinScale > widthMinScale else widthMinScale

        sfactor = random.random() * (scaleMax - scaleMin) + scaleMin  # 0.125 to 1
        scaled = transformed.resize((int(transformed.width * sfactor), 
                                     int(transformed.height * sfactor)))
        xpos = random.randint(0, bg.width - scaled.width)
        ypos = random.randint(0, bg.height - scaled.height)
        bg.paste(scaled, (xpos, ypos), scaled)
        outName = "Test" + "{index:0" + str(formatPlaces) + "d}"
        bg.save(outName.format(index=i) + ".png")
        tilted.close()
        rotated.close()
        scaled.close()
        lighted.close()
        bg.close()








