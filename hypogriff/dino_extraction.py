from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
import cv2
import numpy as np
from PIL import Image
from ray_caster import RayCaster
import time
import torch
from torchvision.ops import box_convert


image_index = 0


class DinoExtraction:
    def __init__(
        self,
        model_config_path="groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_checkpoint_path="weights/groundingdino_swint_ogc.pth"
    ):
        print(f"Loading DINO model: {model_config_path}, {model_checkpoint_path}")
        self.model = load_model(model_config_path, model_checkpoint_path)
        print("DINO model loaded.")

    def prepare_pil_image(self, pil_image):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        rgb_source = pil_image.convert("RGB")
        image_source = np.asarray(rgb_source)
        image, _ = transform(rgb_source, None)
        return image_source, image


    def extract_dino_features(self, pil_image, prompt, box_threshold=0.25, text_threshold=0.25):
        # the input image is a PIL image

        # prepare the image
        image_source, image = self.prepare_pil_image(pil_image)


        # perform DINO grounding extraction
        
        
        start = time.time()

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        end = time.time()

        # print with two decimal places
        print(f"Extracted DINO grounding in {end - start:.2f} seconds.")

        
        # get the dimensions of image
        _, img_height, img_width = image.shape

        boxes = boxes * torch.Tensor([img_width, img_height, img_width, img_height])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidences = logits.numpy()

        # convert the boxes to a dictionary where the key is the phrase and the value is a list of boxes
        phrase_dict = {}
        for box, confidence, phrase in zip(xyxy, confidences, phrases):
            phrase = phrase.strip()
            if phrase not in phrase_dict:
                phrase_dict[phrase] = []

            phrase_dict[phrase].append({
                'x1Rel': box[0] / img_width,
                'y1Rel': box[1] / img_height,
                'x2Rel': box[2] / img_width,
                'y2Rel': box[3] / img_height,
                'c': confidence.item(),
                'rays': [],
            })

        # append another box with 'null' phrase and confidence 1 for the whole image
        phrase_dict['null'] = [{
            'x1Rel': 0,
            'y1Rel': 0,
            'x2Rel': 1,
            'y2Rel': 1,
            'c': 1,
            'rays': [],
        }]

        return phrase_dict
    
    def make_on_message(self):
        this_dino = self

        async def on_message(message, dino_callback=None):
            
            # find the imageData and namedRectangles
    
            if not message:
                return
            
            if 'imageData' not in message:
                print("Error: Could not find image data.")
                return
            
            image_data = message['imageData']

            if 'namedRectangles' not in image_data:
                print("Error: Could not find named rectangles.")
                return
            
            named_rectangles = image_data['namedRectangles']

            left_boom = named_rectangles['left-boom']

            if not left_boom:
                print("Error: Could not find left boom.")
                return
            
            print("Received left boom image.")

            # find the viewInfos in the message
            if 'viewInfos' not in image_data:
                print("Error: Could not find view infos.")
                return

            view_infos = image_data['viewInfos']

            # find the viewInfos for the left and right boom
            left_view_info = None
            right_view_info = None

            for view_info in view_infos:
                if view_info['name'] == 'left-boom':
                    left_view_info = view_info
                elif view_info['name'] == 'right-boom':
                    right_view_info = view_info

            if not left_view_info or not right_view_info:
                print("Error: Could not find view infos for left or right boom.")
                return

            # the images are already decoded, so we can just pass them to extracat_dino_features

            print("Extracting DINO grounding from left boom image.")

            left_dino_prompt = left_view_info['dinoPrompt']
            if not left_dino_prompt:
                print("Error: Could not find DINO prompt for left boom.")
                return
            
            phrase_dict = this_dino.extract_dino_features(left_boom, left_dino_prompt)

            # get the dimensions of the left and right images
            left_width, left_height = left_boom.size

            # print that we are happy
            print("DINO grounding extracted.")

            

            left_fov = left_view_info['fov']
            right_fov = right_view_info['fov']

            left_position = left_view_info['positionBodyFrame']
            right_position = right_view_info['positionBodyFrame']

            left_axis = left_view_info['axisBodyFrame']
            right_axis = right_view_info['axisBodyFrame']

            left_up = left_view_info['upBodyFrame']
            right_up = right_view_info['upBodyFrame']

            left_matrix_world = left_view_info['matrixWorldColumnMajor']
            right_matrix_world = right_view_info['matrixWorldColumnMajor']

            left_projection_matrix_inverse = left_view_info['projectionMatrixInverseColumnMajor']
            right_projection_matrix_inverse = right_view_info['projectionMatrixInverseColumnMajor']

    

            caster = RayCaster(left_matrix_world, left_projection_matrix_inverse, left_width, left_height)

            # iterate over the phrases
            for phrase in phrase_dict:
                # get the list of boxes
                boxes = phrase_dict[phrase]

                # iterate over the boxes
                for box in boxes:
                    # get the box coordinates
                    x1Rel = box['x1Rel']
                    y1Rel = box['y1Rel']
                    x2Rel = box['x2Rel']
                    y2Rel = box['y2Rel']

                    # calculate the relative area of the box
                    areaRel = (x2Rel - x1Rel) * (y2Rel - y1Rel)
                    ray_density = 100
                    n_rays = int(areaRel * ray_density)

                    # at least ten rays
                    n_rays = max(n_rays, 10)

                    for i in range(n_rays):
                        # get a random point in the box
                        xRel = np.random.uniform(x1Rel, x2Rel)
                        yRel = np.random.uniform(y1Rel, y2Rel)

                        # read the color of the pixel
                        color = left_boom.getpixel((xRel * left_width, yRel * left_height))

                        # convert the point to normalized device coordinates
                        xNdc = 2 * xRel - 1
                        yNdc = -2 * yRel + 1

                        # convert the point to a ray
                        (origin, dir) = caster.normalized_device_coords_to_ray(xNdc, yNdc)

                        ray = {
                            # convert to plain python lists
                            'origin': origin.tolist(),
                            'dir': dir.tolist(),
                            'color': [color[0], color[1], color[2]]
                        }

                        # add the ray to the box
                        box['rays'].append(ray)

            if dino_callback:
                # print("Calling stereo callback.")
                # await stereo_callback(point_cloud)
                await dino_callback(phrase_dict, left_matrix_world)



        return on_message




async def on_message(message, dino_callback=None):

    # find the imageData and namedRectangles

    if not message:
        return

    if 'imageData' not in message:
        print("Error: Could not find image data.")
        return

    image_data = message['imageData']

    if 'namedRectangles' not in image_data:
        print("Error: Could not find named rectangles.")
        return

    named_rectangles = image_data['namedRectangles']

    left_boom = named_rectangles['left-boom']

    if not left_boom:
        print("Error: Could not find left boom.")
        return

    print("Received left boom image.")

    # the images are already decoded, so we can just pass them to extract_sift_features

    print("Extracting DINO grounding from left boom image.")
    phrase_dict = extract_dino_features(left_boom)

    # get the dimensions of the left and right images
    left_width, left_height = left_boom.size

    # find the viewInfos in the message
    if 'viewInfos' not in image_data:
        print("Error: Could not find view infos.")
        return

    view_infos = image_data['viewInfos']

    # find the viewInfos for the left and right boom
    left_view_info = None
    right_view_info = None

    for view_info in view_infos:
        if view_info['name'] == 'left-boom':
            left_view_info = view_info
        elif view_info['name'] == 'right-boom':
            right_view_info = view_info

    if not left_view_info or not right_view_info:
        print("Error: Could not find view infos for left or right boom.")
        return

    left_fov = left_view_info['fov']
    right_fov = right_view_info['fov']

    left_position = left_view_info['positionBodyFrame']
    right_position = right_view_info['positionBodyFrame']

    left_axis = left_view_info['axisBodyFrame']
    right_axis = right_view_info['axisBodyFrame']

    left_up = left_view_info['upBodyFrame']
    right_up = right_view_info['upBodyFrame']

    left_matrix_world = left_view_info['matrixWorldColumnMajor']
    right_matrix_world = right_view_info['matrixWorldColumnMajor']

    left_projection_matrix_inverse = left_view_info['projectionMatrixInverseColumnMajor']
    right_projection_matrix_inverse = right_view_info['projectionMatrixInverseColumnMajor']

    

    left_ray_caster = RayCaster(
        left_matrix_world, left_projection_matrix_inverse, left_width, left_height)

    global image_index
    image_index += 1

    def convert_reference_cloud(reference_cloud):
        # convert the reference cloud to a list of points
        points = []

        # reference_cloud is a list of tuples of (direction, depth, error)
        for reference_point in reference_cloud:
            direction = reference_point[0]
            depth = reference_point[1]
            error = reference_point[2]

            # filter out points with depth less than 3
            if depth < 3:
                continue

            # filter out points with error greater than 2
            if error > 2:
                continue

            # compute the point
            point = left_position + direction * depth

            data = [
                point[0],
                point[1],
                point[2]
            ]

            # add the point to the list
            points.append(data)

        return points

    # convert the reference clouds to lists of points
    left_reference_cloud = convert_reference_cloud(left_reference_depths)
    # right_reference_cloud = convert_reference_cloud(right_reference_depths)

    if dino_callback:
        # print("Calling stereo callback.")
        # await stereo_callback(point_cloud)
        await dino_callback(left_reference_cloud, left_matrix_world)
    else:
        print("Error: Stereo callback is None.")
