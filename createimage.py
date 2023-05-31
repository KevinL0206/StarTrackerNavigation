from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import math
import random


# Define class names
constellation = 'Background'
class_names = ['URSA','ORION','Background']
label = 3


# Define input and output folders
input_folder = 'constellations_YOLO_val'

# Define star positions

stars = [[433.8, 212.6],[433.8, 255.9],[354.9, 212.6],[340.5, 255.9],[298.1, 133.1],[270.1, 85.1],[209.4, 79.9]] #URSA

#stars = [(150, 200), (350, 200), (256, 300), (325, 310), (190, 310), (350, 400), (150, 400), (256, 350)] #ORION
#stars = []


# Define image size
img_width = 512
img_height = 512

# Define number of images to generate
num_images = 2

# Define output folder
output_folder = 'constellations_YOLO_val'

# Define rotation angles for augmentation
rotation_angles = [-180,-105, -90,-75, -60,-45,-30, -15,0, 15,30, 45, 60,75, 90, 105, 180]

elevation_angles = [0,0.25, 0.5, 0.75, 1]

# Define noise probability
noise_prob = 0.00
# Define noise pixel size range
min_noise_size = 2
max_noise_size = 4

# Define perspective distortion parameters
max_perspective_distortion = 0.3  # maximum distortion
num_perspective_points = 4  # number of points to generate for perspective distortion

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create un-augmented image
img_orig = Image.new('RGB', (img_width, img_height), (0, 0, 0))
draw = ImageDraw.Draw(img_orig)
for star_num, star in enumerate(stars):
    draw.ellipse((star[0]-10, star[1]-10, star[0]+10, star[1]+10), fill='white')
    #draw.text((star[0]+10, star[1]+10), f'S{star_num+1}', font=ImageFont.truetype('arial.ttf', 20), fill='white')
img_orig.save(os.path.join(output_folder, f'{constellation}_orig_{label}_val.png'))

# Generate images
for i in range(num_images):
    # Create new image object
    img = Image.new('RGB', (img_width, img_height), (0, 0, 0))

    # Draw stars
    draw = ImageDraw.Draw(img)
    for star_num, star in enumerate(stars):
        draw.ellipse((star[0]-10, star[1]-10, star[0]+10, star[1]+10), fill='white')
        #draw.text((star[0]+10, star[1]+10), f'S{star_num+1}', font=ImageFont.truetype('arial.ttf', 20), fill='white')

    # Augment images
    for angle in rotation_angles:

        for elevation in elevation_angles:
            # Create new star positions for each elevation angle
            
            new_stars = []
            for star in stars:
                
                x = star[0] - img_width/2
                y = (star[1] - img_height/2) * elevation
                r = math.sqrt(x**2 + y**2)
                theta = math.atan2(y, x) + math.radians(angle)
                new_x = r * math.cos(theta) + img_width/2
                new_y = r * math.sin(theta) + img_height/2
                new_stars.append((new_x, new_y))

            # Create new image object
            img_rot = Image.new('RGB', (img_width, img_height), (0, 0, 0))

            # Draw stars at new positions
            draw_rot = ImageDraw.Draw(img_rot)
            for star_num, star in enumerate(new_stars):
                draw_rot.ellipse((star[0]-10, star[1]-10, star[0]+10, star[1]+10), fill='white')
                #draw_rot.text((star[0]+10, star[1]+10), f'S{star_num+1}', font=ImageFont.truetype('arial.ttf', 20), fill='white')

            img_rot = Image.fromarray(np.uint8(np.clip(np.array(img_rot), 0, 255)))
            # Add brightness variation
            brightness = np.random.uniform(0.5, 1.5)
            img_rot = Image.fromarray(np.uint8(np.clip(np.array(img_rot) * brightness, 0, 255)))

            # Add noise pixels of various sizes and brightness
            for x in range(img_width):
                for y in range(img_height):
                    if np.random.random() < noise_prob:
                        noise_size = np.random.randint(min_noise_size, max_noise_size)
                        min_brightness = 128
                        max_brightness = 255
                        noise_color = tuple([random.randint(min_brightness, max_brightness)]*3)
                        x1 = max(0, x - noise_size // 2)
                        x2 = min(img_width, x + noise_size // 2 + 1)
                        y1 = max(0, y - noise_size // 2)
                        y2 = min(img_height, y + noise_size // 2 + 1)
                        xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
                        for u in range(x1, x2):
                            for v in range(y1, y2):
                                d = math.sqrt((u - xc)**2 + (v - yc)**2)
                                if d <= noise_size // 2:
                                    img_rot.putpixel((u, v), noise_color)
                                else:
                                    img_rot.putpixel((u, v), (0, 0, 0))


            # Save image with elevation angle in filename
            img_rot.save(os.path.join(output_folder, f'{constellation}_{i+1}_angle_{angle}_elevation_{elevation}_{label}_val.png'))

       
        
# Define label folder
label_folder = 'constellations_labels'

# Create label folder if it doesn't exist
if not os.path.exists(label_folder):
    os.makedirs(label_folder)

# Define label file format
label_format = '{}.txt'

# Generate YOLO labels
for i in range(num_images):
    
    for angle in rotation_angles:
        
        for elevation in elevation_angles:
            # Load image
            img_file = os.path.join(output_folder, f'{constellation}_{i+1}_angle_{angle}_elevation_{elevation}_{label}_val.png')
            img = cv2.imread(img_file)
            # Load original image for size reference
            img_orig_file = os.path.join(input_folder, f'{constellation}_orig_{label}_val.png')
            img_orig = cv2.imread(img_orig_file)
            orig_height, orig_width, _ = img_orig.shape

            new_stars = []
            for star in stars:

                x = star[0] - img_width/2
                y = (star[1] - img_height/2) * elevation
                r = math.sqrt(x**2 + y**2)
                theta = math.atan2(y, x) + math.radians(angle)
                new_x = r * math.cos(theta) + img_width/2
                new_y = r * math.sin(theta) + img_height/2
                
                new_stars.append((new_x, new_y))

            # Compute bounding box for entire constellation
            min_x = min(star[0] for star in new_stars) - 15
            max_x = max(star[0] for star in new_stars) + 15
            min_y = min(star[1] for star in new_stars) - 15
            max_y = max(star[1] for star in new_stars) + 15

            # Define label file
            label_file = os.path.join(label_folder, label_format.format(f'{constellation}_{i+1}_angle_{angle}_elevation_{elevation}_{label}_val'))

            # Open label file
            with open(label_file, 'w') as f:
                # Write label for entire constellation
                x_center = (min_x + max_x) / 2 / orig_width
                y_center = (min_y + max_y) / 2 / orig_height
                width = (max_x - min_x) / orig_width
                height = (max_y - min_y) / orig_height
                class_id = class_names.index(constellation)
                f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')
       

"""
# Visualize bounding boxes
for i in range(num_images):
    for angle in rotation_angles:
        for elevation in elevation_angles:

            # Load image and label
            img_file = os.path.join(output_folder, f'{constellation}_{i+1}_angle_{angle}_elevation_{elevation}_{label}.png')
            img = cv2.imread(img_file)
            label_file = os.path.join(label_folder, label_format.format(f'{constellation}_{i+1}_angle_{angle}_elevation_{elevation}_{label}'))
            
            # Read label file
            with open(label_file, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = line.strip().split()
                    class_id = int(class_id)
                    x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
                    
                    # Convert coordinates to image coordinates
                    x_min = int((x_center - width/2) * img.shape[1])
                    y_min = int((y_center - height/2) * img.shape[0])
                    x_max = int((x_center + width/2) * img.shape[1])
                    y_max = int((y_center + height/2) * img.shape[0])
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Display image with bounding boxes
            cv2.imshow(f'constellation_{i+1}_angle_{angle}_elevation_{elevation}', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
"""
