from flask import Flask, request, render_template, jsonify, send_from_directory
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
from skimage.draw import line
import os
import csv
import uuid

line_cache = {}  # To store line coordinates between dots
line_score_cache = {}  # To store sum of pixel values for each line

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def svg_to_dots(svg_path, num_dots=100, image_size=(500, 500)):
    paths, attributes = svg2paths(svg_path)
    points = []

    for path in paths:
        for i in range(num_dots):
            point = path.point(i / num_dots)
            points.append((point.real, point.imag))

    points = np.array(points)
    
    # Normalize points to fit the image size
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    points = (points - [min_x, min_y]) / ([max_x - min_x, max_y - min_y])
    points = points * (np.array(image_size) - 1)
    
    # Generate sequential IDs for each point
    point_data = [{'id': i, 'x': float(p[0]), 'y': float(p[1])} for i, p in enumerate(points)]

    # Save the points to a CSV file
    csv_path = os.path.join(RESULT_FOLDER, 'points.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['id', 'x', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in point_data:
            writer.writerow(data)
    
    return points, point_data

def create_dots_image(dots, output_path, image_size=(500, 500), dot_radius=3):
    # Create a blank image with transparent background
    image = Image.new('RGBA', image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    # Draw each dot
    for dot in dots:
        x, y = int(dot[0]), int(dot[1])
        draw.ellipse((x-dot_radius, y-dot_radius, x+dot_radius, y+dot_radius), fill='black')

    # Save the image with DPI set to 100
    image.save(output_path, dpi=(100, 100))

def crop_image(image_path, crop_x, crop_y, crop_width, crop_height):
    image = Image.open(image_path)
    cropped_image = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
    cropped_image_path = os.path.join(UPLOAD_FOLDER, 'cropped_image.jpg')
    cropped_image.save(cropped_image_path)
    return cropped_image_path

def cache_line_coordinates(dots, resolution):
    global line_cache
    line_cache = {}
    num_pins = len(dots)
    for start_idx in range(num_pins):
        for end_idx in range(start_idx + 1, num_pins):
            start, end = dots[start_idx], dots[end_idx]
            rr, cc = line(int(start[1]), int(start[0]), int(end[1]), int(end[0]))
            rr = np.clip(rr, 0, resolution[1] - 1)
            cc = np.clip(cc, 0, resolution[0] - 1)
            
            line_cache[(start_idx, end_idx)] = (rr, cc)
            line_cache[(end_idx, start_idx)] = (rr, cc)

def create_string_art(jpg_path, svg_path, output_path, num_dots=100, num_lines=1000, resolution=(500, 500),
                      forbidden_neighbors=0, power=255, line_intensity=255, line_score_method='mean'):
    # Load the image
    image = Image.open(jpg_path).convert('L')
    image = image.resize(resolution)
    image = np.array(image, dtype=np.uint8)
    image = image / 255.0  # Normalize to 0-1 range

    # Convert SVG to dots
    dots, _ = svg_to_dots(svg_path, num_dots=num_dots, image_size=resolution)

    # Cache line coordinates
    cache_line_coordinates(dots, resolution)

    # Initialize the canvas
    canvas = np.ones(resolution, dtype=np.float32)
    line_intensity /= 255.0
    power /= 255.0

    # Set up forbidden neighbors
    forbidden_pairs = set()
    for i in range(len(dots)):
        for j in range(1, forbidden_neighbors + 1):
            if i + j < len(dots):
                forbidden_pairs.add((i, i + j))
                forbidden_pairs.add((i + j, i))

    # Find the darkest point in the image to start the string art
    start_point = np.unravel_index(np.argmin(image, axis=None), image.shape)
    start_idx = np.argmin(np.linalg.norm(dots - start_point[::-1], axis=1))

    current_idx = start_idx
    used_ids = []  # Track used point IDs

    # Draw lines
    for _ in range(num_lines):
        best_score = float('inf')
        best_end_idx = None
        best_rr, best_cc = None, None

        # Iterate over all possible end points
        for end_idx in range(len(dots)):
            if end_idx == current_idx or (current_idx, end_idx) in forbidden_pairs:
                continue

            # Retrieve cached line coordinates
            if (current_idx, end_idx) in line_cache:
                rr, cc = line_cache[(current_idx, end_idx)]
            else:
                continue

            # Calculate line score based on method
            if line_score_method == 'mean':
                line_score = np.mean(image[rr, cc])
            elif line_score_method == 'sum':
                line_score = np.sum(image[rr, cc])

            # Track the best scoring line
            if line_score < best_score:
                best_score = line_score
                best_end_idx = end_idx
                best_rr, best_cc = rr, cc

        if best_end_idx is None:
            break

        # Draw line on canvas
        canvas[best_rr, best_cc] = np.maximum(0, canvas[best_rr, best_cc] - line_intensity)

        # Modify the image based on the drawn line
        image[best_rr, best_cc] += power * (1 - image[best_rr, best_cc])

        # Update used_ids and current index
        used_ids.append(current_idx)
        current_idx = best_end_idx

    # Save the used IDs to a file
    used_ids_path = os.path.join(RESULT_FOLDER, 'used_ids.txt')
    with open(used_ids_path, 'w') as f:
        f.write(','.join(map(str, used_ids)))

    # Save the final string art image
    plt.imsave(output_path, canvas, cmap='gray', vmin=0, vmax=1, dpi=100)
    
def create_color_string_art(jpg_path, svg_path, output_path, num_dots=100, num_lines=1000, resolution=(500, 500),
                            forbidden_neighbors=0, power=255, line_intensity=255, line_score_method='mean'):
    # Load the image in RGB
    image = Image.open(jpg_path).convert('RGB')
    image = image.resize(resolution)
    image = np.array(image, dtype=np.float64)  # Convert to float32 for easier manipulation
    image /= 255.0  # Normalize to 0-1 range

    # Convert RGB to CMY (using inverted RGB)
    cyan_channel = 1 - image[:, :, 0]  # Cyan is the opposite of Red
    magenta_channel = 1 - image[:, :, 1]  # Magenta is the opposite of Green
    yellow_channel = 1 - image[:, :, 2]  # Yellow is the opposite of Blue

    # Convert SVG to dots
    dots, _ = svg_to_dots(svg_path, num_dots=num_dots, image_size=resolution)

    # Cache line coordinates
    cache_line_coordinates(dots, resolution)

    # Initialize the canvas for drawing in RGB format (White background)
    canvas = np.ones((*resolution, 3), dtype=np.float64)

    # Set up forbidden neighbors
    forbidden_pairs = set()
    for i in range(len(dots)):
        for j in range(1, forbidden_neighbors + 1):
            if i + j < len(dots):
                forbidden_pairs.add((i, i + j))
                forbidden_pairs.add((i + j, i))

    used_ids_dict = {'#00FFFF': [], '#FF00FF': [], '#FFFF00': []}  # Store used IDs for Cyan, Magenta, and Yellow

    def draw_lines(color_channel, color, color_name):
        # Find the darkest point in the color channel to start the string art
        start_point = np.unravel_index(np.argmax(color_channel, axis=None), color_channel.shape)
        start_idx = np.argmin(np.linalg.norm(dots - start_point[::-1], axis=1))

        current_idx = start_idx

        # Draw lines
        for _ in range(num_lines):
            best_score = float('-inf')  # We are looking for the highest score (darkest point)
            best_end_idx = None
            best_rr, best_cc = None, None

            # Iterate over all possible end points
            for end_idx in range(len(dots)):
                if end_idx == current_idx or (current_idx, end_idx) in forbidden_pairs:
                    continue

                # Retrieve cached line coordinates
                if (current_idx, end_idx) in line_cache:
                    rr, cc = line_cache[(current_idx, end_idx)]
                else:
                    continue

                # Calculate line score based on method
                if line_score_method == 'mean':
                    line_score = np.mean(color_channel[rr, cc])
                elif line_score_method == 'sum':
                    line_score = np.sum(color_channel[rr, cc])

                # Track the best scoring line (highest score for darkest path)
                if line_score > best_score:
                    best_score = line_score
                    best_end_idx = end_idx
                    best_rr, best_cc = rr, cc

            if best_end_idx is None:
                break

            # Draw line on canvas for the specific color
            for i in range(3):
                if i == color:  # Only adjust the specific color channel
                    canvas[best_rr, best_cc, i] = np.maximum(0, canvas[best_rr, best_cc, i] - line_intensity / 255.0)

            # Modify the color channel based on the drawn line
            color_channel[best_rr, best_cc] = np.clip(color_channel[best_rr, best_cc] - power / 255.0, 0, 1)

            # Update current index and used IDs list for the specific color
            used_ids_dict[color_name].append(current_idx)
            current_idx = best_end_idx

    # Draw lines for Cyan
    draw_lines(cyan_channel, color=0, color_name='#00FFFF')  # Cyan
    # Draw lines for Magenta
    draw_lines(magenta_channel, color=1, color_name='#FF00FF')  # Magenta
    # Draw lines for Yellow
    draw_lines(yellow_channel, color=2, color_name='#FFFF00')  # Yellow

    # Save the used IDs to a file, separating colors into paragraphs
    used_ids_path = os.path.join(RESULT_FOLDER, 'used_ids.txt')
    with open(used_ids_path, 'w') as f:
        for color, ids in used_ids_dict.items():
            f.write(f'{color}:\n')
            f.write(','.join(map(str, ids)) + '\n\n')

    # Save the final string art image
    plt.imsave(output_path, canvas, vmin=0, vmax=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-dots', methods=['POST'])
def generate_dots():
    svg_file = request.files.get('svgUpload')
    num_dots = int(request.form.get('numDots', 100))
    resolution = tuple(map(int, request.form.get('resolution', '500,500').split(',')))

    if svg_file:
        svg_path = os.path.join(UPLOAD_FOLDER, 'outline.svg')
        dots_image_path = os.path.join(RESULT_FOLDER, 'dots_image.png')
        points_csv_path = os.path.join(RESULT_FOLDER, 'points.csv')

        svg_file.save(svg_path)

        # Generate and save the dots image
        dots, _ = svg_to_dots(svg_path, num_dots=num_dots, image_size=resolution)
        create_dots_image(dots, dots_image_path, image_size=resolution)

        return jsonify({
            'dots_image_url': f'/results/dots_image.png',
            'points_csv_url': f'/results/points.csv'
        })

    return 'Failed to upload SVG file', 400

@app.route('/upload', methods=['POST'])
def upload():
    svg_file = request.files.get('svgUpload')
    jpg_file = request.files.get('jpgUpload')
    num_dots = int(request.form.get('numDots', 100))
    num_lines = int(request.form.get('numLines', 1000))
    resolution = tuple(map(int, request.form.get('resolution', '500,500').split(',')))
    offset_x = float(request.form.get('offsetX', 0))
    offset_y = float(request.form.get('offsetY', 0))
    scale = float(request.form.get('scale', 1))
    forbidden_neighbors = int(request.form.get('forbiddenNeighbors', 0))
    line_intensity = int(request.form.get('line_intensity', 255))
    power = int(request.form.get('power', 255))  # Default to 255
    line_score_method = request.form.get('lineScoreMethod', 'mean')  # Default to 'mean'
    type_art = request.form.get('typeArt', 'blackWhite')  # Default to 'blackWhite'

    crop_x = int(request.form.get('cropX', 0))
    crop_y = int(request.form.get('cropY', 0))
    crop_width = int(request.form.get('cropWidth', resolution[0]))
    crop_height = int(request.form.get('cropHeight', resolution[1]))

    # Retrieve brightness (luminosity) and contrast values from the form
    brightness = int(request.form.get('luminosity', 0))  # Default to 0 (no change)
    contrast = int(request.form.get('contrast', 0))  # Default to 0 (no change)

    if svg_file and jpg_file:
        svg_path = os.path.join(UPLOAD_FOLDER, 'outline.svg')
        jpg_path = os.path.join(UPLOAD_FOLDER, 'image.jpg')
        result_path = os.path.join(RESULT_FOLDER, 'string_art.png')
        used_ids_path = os.path.join(RESULT_FOLDER, 'used_ids.txt')

        svg_file.save(svg_path)
        jpg_file.save(jpg_path)

        # Crop the image before creating string art
        cropped_img_path = crop_image(jpg_path, crop_x, crop_y, crop_width, crop_height)
        cropped_img = Image.open(cropped_img_path)

        # Apply brightness and contrast adjustments
        # Brightness and contrast range is from -100 to 100, map it to Pillow's expected range
        brightness_factor = 1 + (brightness / 100.0)  # Maps -100 to +100 to 0.0 to 2.0 (1.0 means no change)
        contrast_factor = 1 + (contrast / 100.0)      # Maps -100 to +100 to 0.0 to 2.0 (1.0 means no change)

        # Apply brightness adjustment
        enhancer_brightness = ImageEnhance.Brightness(cropped_img)
        cropped_img = enhancer_brightness.enhance(brightness_factor)

        # Apply contrast adjustment
        enhancer_contrast = ImageEnhance.Contrast(cropped_img)
        cropped_img = enhancer_contrast.enhance(contrast_factor)

        # Save the adjusted image to use it for string art
        adjusted_img_path = os.path.join(UPLOAD_FOLDER, 'adjusted_image.jpg')
        cropped_img.save(adjusted_img_path)

        # Redirect to the appropriate function based on typeArt value
        if type_art == 'blackWhite':
            create_string_art(adjusted_img_path, svg_path, result_path, 
                              num_dots=num_dots, num_lines=num_lines, resolution=resolution, 
                              forbidden_neighbors=forbidden_neighbors, power=power, 
                              line_intensity=line_intensity, line_score_method=line_score_method)
        elif type_art == 'color':
            create_color_string_art(adjusted_img_path, svg_path, result_path, 
                                    num_dots=num_dots, num_lines=num_lines, resolution=resolution, 
                                    forbidden_neighbors=forbidden_neighbors, power=power, 
                                    line_intensity=line_intensity, line_score_method=line_score_method)

        return jsonify({
            'string_art_url': f'/results/string_art.png',
            'used_ids_url': f'/results/used_ids.txt'
        })

    return 'Failed to upload files', 400

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
