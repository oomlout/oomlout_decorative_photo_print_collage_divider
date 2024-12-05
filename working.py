import os
import random
import math
import numpy as np
import cv2
from shapely.geometry import box
from shapely.affinity import rotate
import yaml

def main(**kwargs):
    file_source = kwargs.get('file_source', 'image\\source.jpg')
    directory_output_base = kwargs.get('directory_output_base', 'output')
    file_source_just_name = os.path.basename(file_source)
    #remove file extension
    file_source_just_name = os.path.splitext(file_source_just_name)[0]
    directory_output = os.path.join(directory_output_base, file_source_just_name)
    if not os.path.exists(directory_output):
        os.makedirs(directory_output)

    #copy source file in to the output director as source.jpg
    file_source_output = os.path.join(directory_output, 'source.jpg')
    if not os.path.exists(file_source_output):
        print(f"Copying source file to {file_source_output}")
        os.system(f"copy {file_source} {file_source_output}")

    large_width = 600
    large_height = 400
    small_width = 150
    small_height = 100
    number_of_rectangles = 25

    file_position_yaml = os.path.join(directory_output, 'positions.yaml')

    if os.path.exists(file_position_yaml):
        with open(file_position_yaml, 'r') as file:
            print("Loading positions from file")
            positions = yaml.load(file, Loader=yaml.FullLoader)
    else:
        print("Generating positions")
        positions = fill_large_rectangle(large_width, large_height, small_width, small_height, number_of_rectangles)
        #save positions in yaml to poistion.yaml in output directory    
        with open(os.path.join(directory_output, 'positions.yaml'), 'w') as file:
            yaml.dump(positions, file)

    # Plot positions in an image where each rectangle is a random color then save as layout.png
    plot_positions(positions, large_width, large_height, small_width, small_height, os.path.join(directory_output, 'layout.png'))

    create_photo_tiles(file_source, positions, large_width, large_height, small_width, small_height, directory_output)


def create_photo_tiles(file_source, positions, large_width, large_height, small_width, small_height, directory_output):
    
    print("Creating photo tiles")
    directory_tile = os.path.join(directory_output, 'tile')
    if not os.path.exists(directory_tile):
        os.makedirs(directory_tile)
    img = cv2.imread(file_source)
    pixel_width = img.shape[1]
    pixel_height = img.shape[0]
    scaling_factor_width = pixel_width / large_width
    scaling_factor_height = pixel_height / large_height

    #add a 500 pixel pad around the image make it black
    pad = 500
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    


    #test to see they are similar
    print(f"Scaling factor width: {scaling_factor_width}")
    print(f"Scaling factor height: {scaling_factor_height}")
    if scaling_factor_width != scaling_factor_height:
        print("Scaling factors are not the same, input image needs to be the right ratio")
        return
    scaling_factor = scaling_factor_width

    for i, pos in enumerate(positions):
        print(f"    Creating tile {i}")
        x, y, rotation = pos
        tile_width = int(small_width * scaling_factor)
        tile_height = int(small_height * scaling_factor)
        tile_x = x * scaling_factor
        tile_y = y * scaling_factor
        tile_x_real = tile_x + pad
        tile_y_real = tile_y + pad
        tile = crop_rotated_rectangle(img, tile_x_real, tile_y_real, tile_width, tile_height, rotation)
        
        #if tile is less than tile_width wide add it as black border
        if tile.shape[1] < tile_width:
            border = np.zeros((tile.shape[0], tile_width - tile.shape[1], 3), np.uint8)
            border.fill(0)
            tile = np.concatenate((tile, border), axis=1)
        #if tile is less than tile_height high add it as black border
        if tile.shape[0] < tile_height:
            border = np.zeros((tile_height - tile.shape[0], tile.shape[1], 3), np.uint8)
            border.fill(0)
            tile = np.concatenate((tile, border), axis=0)

            
        
        #crop out the middle tile_width x tile_height
        start_x = (tile.shape[1] - tile_width) // 2
        start_y = (tile.shape[0] - tile_height) // 2
        tile = tile[start_y:start_y + tile_height, start_x:start_x + tile_width]
              


        
    


        file_name = os.path.join(directory_tile, f'tile_{i}.png')
        print(f"        Saving tile {i} to {file_name}")
        cv2.imwrite(file_name, tile)

def old_1_crop_rotated_rectangle(img, x, y, width, height, angle):
    rect = box(x, y, x + width, y + height)
    rotated_rect = rotate(rect, angle, origin='center')
    coords = list(rotated_rect.exterior.coords)
    pts = [(int(pt[0]), int(pt[1])) for pt in coords[:-1]]  # Exclude the last point because it's the same as the first

    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.array(pts)], color=(255))

    


    return cv2.bitwise_and(img, img, mask=mask)

def crop_rotated_rectangle(img, x, y, width, height, angle):
    rect = box(x, y, x + width, y + height)
    rotated_rect = rotate(rect, angle, origin='center')
    coords = list(rotated_rect.exterior.coords)
    pts = [(int(pt[0]), int(pt[1])) for pt in coords[:-1]]  # Exclude the last point because it's the same as the first

    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.array(pts)], color=(255))

    img = cv2.bitwise_and(img, img, mask=mask)

    
    #crop the image to the counding box of the rotated rectangle
    x, y, w, h = cv2.boundingRect(np.array(pts))
    img = img[y:y+h, x:x+w]

    
    #remove the angle roataion
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    #remove black border
    


    return img


def plot_positions(positions, large_width, large_height, small_width, small_height, file_output):
    img = np.zeros((large_height, large_width, 3), np.uint8)
    img.fill(255)

    for pos in positions:
        x, y, rotation = pos
        draw_rotated_rectangle(img, x, y, small_width, small_height, rotation)

    cv2.imwrite(file_output, img)

def draw_rotated_rectangle(img, x, y, width, height, angle):
    rect = box(x, y, x + width, y + height)
    rotated_rect = rotate(rect, angle, origin='center')
    coords = list(rotated_rect.exterior.coords)
    pts = [(int(pt[0]), int(pt[1])) for pt in coords[:-1]]  # Exclude the last point because it's the same as the first
    
    #mode = "outline"
    mode = "fill"
    if mode == "fill":
        cv2.fillPoly(img, [np.array(pts)], color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    else:
        cv2.polylines(img, [np.array(pts)], isClosed=True, color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), thickness=2)

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def distance_to_edges(point, width, height):
    x, y = point
    return min(x, width - x, y, height - y)

def distance_to_points(point, points):
    x1, y1 = point
    #make this more lines
    distances = [calculate_distance((x1, y1), (x2, y2)) for x2, y2, _ in points]
    return min(distances)

def find_furthest_point(points, width, height):
    max_distance = -1
    furthest_point = None
    scalar = 10
    for x in range(1,width + 1,scalar):
        for y in range(1,height + 1,scalar):
            point = (x*scalar, y*scalar)
            dist_to_edges = distance_to_edges(point, width, height)
            dist_to_points = distance_to_points(point, points)
            distance = min(dist_to_edges, dist_to_points)
            if distance > max_distance:
                max_distance = distance
                furthest_point = point
    return furthest_point

def generate_random_rotation():
    return random.randint(0, 180) 

def fill_large_rectangle(large_width, large_height, small_width, small_height, number_of_rectangles=10):
    positions = []
    x, y = 0, 0

    random_x = random.randint(0, large_width)
    random_y = random.randint(0, large_height)
    positions.append((random_x, random_y, generate_random_rotation()))

    for i in range(1, number_of_rectangles):        
        furthest_point = find_furthest_point(positions, large_width, large_height)
        print(f"Furthest point: {furthest_point}")
        positions.append((furthest_point[0], furthest_point[1], generate_random_rotation()))

    return positions

if __name__ == '__main__':
    kwargs = {}
    file_source = "image\\source.jpg"
    #file_source = "image\\source_1.jpg"
    kwargs['file_source'] = file_source
    directory_output_base = "output"
    kwargs['directory_output_base'] = directory_output_base
    main(**kwargs)