import os
import random
import rasterio
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from albumentations import Compose, Lambda, GridDistortion, ElasticTransform, Flip, Affine
import cv2

def filter_image_with_mask(block, block_image):
    """
    Применяет маску к изображению.

    Параметры:
    - block (numpy.ndarray): Маска блока.
    - block_image (numpy.ndarray): Изображение блока.

    Возвращает:
    - numpy.ndarray: Отфильтрованное изображение.
    """
    block_local = block.copy()

    # Изменение размера маски, если необходимо
    if block_local.shape[:2] != block_image.shape[:2]:
        block_local = cv2.resize(block_local, (block_image.shape[1], block_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Преобразование маски в бинарный формат
    block_local[block_local >= 1] = 1

    # Создание 3D-маски для применения к изображению
    mask_3d = np.repeat(block_local[:, :, np.newaxis], 3, axis=2)

    # Применение маски к изображению
    filtered_image = block_image * mask_3d

    return filtered_image

def image_verification(img, mask):
    """
    Checks the size of the image and the mask.,
    as well as the number of non-zero pixels in the mask.
    Parameters:
        - img (numpy.ndarray): Image.
        - mask (numpy.ndarray): Mask.

    Returns:
    - bool or str: True if the check is passed, "Empty" if there are few non-zero pixels, otherwise False.
    """
    if img.shape[:2] != mask.shape[:2]:
        return False   
    if img.shape[1] < 100 or img.shape[0] < 100:
        return False

    non_zero_count = np.count_nonzero(mask)

    if non_zero_count > 50:
        return True
    elif non_zero_count < 50:
        return "Пусто"

    return False

transform = Compose([
    Lambda(image=lambda image, **kwargs: mix_channel(image, **kwargs), mask=lambda mask, **kwargs: mask),
    GridDistortion(p=0.5),
    ElasticTransform(p=0.5),
    Flip(p=0.5),
    Affine(
        rotate=0,  
        p=0.5  
    )
])

def mix_channel(image, substitution_prob=0.5, **kwargs):
    """
    Randomly replaces the channels of the image.

    Parameters:
    - image (numpy.ndarray): Input image.
    - substitution_prob (float): The probability of replacing each channel.

    Returns:
    - numpy.ndarray: Modified image.
    """
    if np.random.rand() < kwargs.get('apply_prob', 0.4):
        num_channels = image.shape[2]
        new_image = image.copy()
        for c in range(num_channels):
            if np.random.rand() < substitution_prob:
                substitute_channel = np.random.randint(num_channels)
                new_image[:, :, c] = image[:, :, substitute_channel]
        return new_image
    return image

class MaskClassifier:
    def __init__(self, mask_files, image_files, path, objects_to_add=5, use_albumentations=False, block_size=512):
        """

        Parameters:
            - mask_files (list): A list of paths to mask files.
            - image_files (list): A list of paths to image files.
            - path (str): The path to the working directory.
            - objects_to_add (int): The number of objects to add during augmentation.
            - use_albumentations (bool): Whether to use augmentations with albumentations.
            - block_size (int): The size of the image splitting block.

        """
        self.path = path
        self.id = 0
        self.mask_files = mask_files
        self.image_files = image_files
        self.objects_to_add = objects_to_add
        self.use_albumentations = use_albumentations
        self.block_size = block_size
        self.initialize_directories()
        if self.use_albumentations:
            self.augmentation_pipeline = self.get_augmentation_pipeline()

    def get_augmentation_pipeline(self):
        """
        Creates an augmentation pipeline using albumentations.

        Returns:
        - albumentations.Compose: An augmentation pipeline.
        """
        return Compose([
            Lambda(image=mix_channel, mask=lambda mask, **kwargs: mask),
            GridDistortion(p=0.5),
            ElasticTransform(p=0.5),
            Flip(p=0.5),
            Affine(
                rotate=0,  
                p=0.5  
            )
        ])

    def initialize_directories(self):
        """
        Creates the necessary directories to save the processed images and masks.     
        """
        directories = ['images', 'masks', 'cropped_images', 'cropped_masks', 'empty_images', 'empty_masks']
        for dir_name in directories:
            os.makedirs(os.path.join(self.path, dir_name), exist_ok=True)

    def apply_bounding_box_and_save(self, mask, object_image, original_image, row_offset, col_offset, max_dim=200, min_area=1300):
        """
        Applies a bounding rectangle to an object and saves images and masks.

        Parameters:
        - mask (numpy.ndarray): The mask of the object.
        - object_image (numpy.ndarray): Image of the object.
        - original_image (numpy.ndarray): The original image.
        - row_offset (int): Row offset.
        - col_offset (int): Column offset.
        - max_dim (int): The maximum block size.
        - min_area (int): The minimum area to save the block.
        """
        non_zero_indices = np.argwhere(object_image[:, :, 0] != 0)
        if non_zero_indices.size == 0:
            return

        x_min, y_min = non_zero_indices.min(axis=0)
        x_max, y_max = non_zero_indices.max(axis=0)
        if y_max - y_min < 50 or x_max - x_min < 50:
            return

        cropped_image = object_image[x_min:x_max+1, y_min:y_max+1].astype(np.uint8)
        mask_to_save = mask[x_min:x_max+1, y_min:y_max+1].astype(np.uint8)

        processed_img_path = os.path.join(self.path, 'cropped_images', f'{self.id}.png')
        Image.fromarray(cropped_image).convert('RGB').save(processed_img_path)
        Image.fromarray(mask_to_save).save(os.path.join(self.path, 'cropped_masks', f'{self.id}.png'))

        values, counts = np.unique(mask, return_counts=True)
        if len(values) <= 2 and (1 in values):
            return

        for i in range(x_min, x_max, max_dim):
            for j in range(y_min, y_max, max_dim):
                sub_x_max = min(i + max_dim, x_max)
                sub_y_max = min(j + max_dim, y_max)
                sub_image = object_image[i:sub_x_max+1, j:sub_y_max+1].astype(np.uint8)
                sub_mask = mask[i:sub_x_max+1, j:sub_y_max+1].astype(np.uint8)

                if np.count_nonzero(sub_mask) > min_area:
                    transformed = transform(image=sub_image, mask=sub_mask)
                    transformed_image = transformed['image']
                    transformed_mask = transformed['mask']

                    image_path = os.path.join(self.path, 'cropped_images', f'{self.id}_{i}_{j}.png')
                    mask_path = os.path.join(self.path, 'cropped_masks', f'{self.id}_{i}_{j}.png')
                    Image.fromarray(transformed_image).convert('RGB').save(image_path)
                    Image.fromarray(transformed_mask).save(mask_path)

    def save_block(self, image, mask, row, col, empty):
        """
        Saves the image block and the corresponding mask.

        Parameters:
        - image (numpy.ndarray): The image of the block.
        - mask (numpy.ndarray): The block mask.
        - row (int): Row offset.
        - col (int): Column offset.
        - empty (bool): A flag indicating whether the block is empty.
        """
        directory = 'empty_images' if empty else 'images'
        mask_directory = 'empty_masks' if empty else 'masks'

        image_path = os.path.join(self.path, directory, f'{self.id}_block_{row}_{col}.png')
        mask_path = os.path.join(self.path, mask_directory, f'{self.id}_block_{row}_{col}.png')
        # print(image_path)
        # print(mask_path)
        Image.fromarray(image).convert('RGB').save(image_path)
        Image.fromarray(mask).save(mask_path)
 
    def process_block(self, block_image, mask_block, row, col):
        """
        Processes a separate block of images and masks.

        Parameters:
        - block_image (numpy.ndarray): Image of the block.
        - mask_block (numpy.ndarray): The block mask.
        - row (int): Row offset.
        - col (int): Column offset.
        """
        check = image_verification(block_image, mask_block)
        if check == True:
            self.save_block(block_image, mask_block, row, col, False)
            binary_clean_image = filter_image_with_mask(mask_block, block_image)
            self.apply_bounding_box_and_save(mask_block, binary_clean_image, block_image, row, col)
        elif check == "Пусто":
            self.save_block(block_image, mask_block, row, col, True)
        # Если check == False, ничего не делаем
        self.id += 1

    def classify_and_save_blocks(self):
        """
        Splits images into blocks and processes each block.       

        """
        for image_file, mask_file in zip(self.image_files, self.mask_files):
            with rasterio.open(image_file) as src_image, rasterio.open(mask_file) as src_mask:
                image_data = src_image.read([1, 2, 3])
                mask_data = src_mask.read(1)

                # Преобразование масок: 0 -> 1, 255 -> 0
                mask_data = np.where(mask_data == 0, 1, mask_data)
                mask_data = np.where(mask_data == 255, 0, mask_data)

                original_image = np.transpose(image_data, (1, 2, 0))
                for row in range(0, original_image.shape[0], self.block_size):
                    for col in range(0, original_image.shape[1], self.block_size):
                        block_image = original_image[row:row+self.block_size, col:col+self.block_size]
                        mask_block = mask_data[row:row+self.block_size, col:col+self.block_size]
                        self.process_block(block_image, mask_block, row, col)

    def augment_images(self, path):
        """
        Applies augmentation to images and masks.
        Parameters:
        - path (str): The path to the working directory..
        """
        all_cropped_images = [
            os.path.join(path, 'cropped_images', f) for f in os.listdir(os.path.join(path, 'cropped_images'))
        ]

        for img_dir in ["images", "empty_images"]:
            for image_file in os.listdir(f"{path}/{img_dir}"):
                original_image_path = os.path.join(path, img_dir, image_file)
                original_image = Image.open(original_image_path)
                original_image_np = np.array(original_image)

                if img_dir == 'images':
                    mask_path = os.path.join(path, 'masks', image_file.replace('rgb', 'mask'))
                    mask = Image.open(mask_path)
                    mask_np = np.array(mask)
                else:
                    mask_np = np.zeros_like(original_image_np[:, :, 0])

                random.shuffle(all_cropped_images)
                cropped_images_paths = all_cropped_images.copy()
                objects_added = 0

                while objects_added < self.objects_to_add and cropped_images_paths:
                    cropped_image_path = cropped_images_paths.pop(0)
                    cropped_image = Image.open(cropped_image_path)
                    cropped_image_np = np.array(cropped_image)

                    cropped_mask_path = cropped_image_path.replace('cropped_images', 'cropped_masks')
                    cropped_mask = Image.open(cropped_mask_path)
                    cropped_mask_np = np.array(cropped_mask)

                    if cropped_image_np.shape[0] > original_image_np.shape[0] or cropped_image_np.shape[1] > original_image_np.shape[1]:
                        continue

                    for attempt in range(100):
                        y_offset = random.randint(0, original_image_np.shape[0] - cropped_image_np.shape[0])
                        x_offset = random.randint(0, original_image_np.shape[1] - cropped_image_np.shape[1])

                        overlay_area = mask_np[y_offset:y_offset + cropped_image_np.shape[0], x_offset:x_offset + cropped_image_np.shape[1]]
                        if np.any(overlay_area > 0):
                            continue

                        overlay_shape = overlay_area.shape[:2]
                        cropped_mask_resized = cropped_mask_np[:overlay_shape[0], :overlay_shape[1]]
                        mask_overlay = np.expand_dims((cropped_mask_resized > 0), axis=-1)

                        original_image_np[y_offset:y_offset + overlay_shape[0], x_offset:x_offset + overlay_shape[1]] = np.where(
                            mask_overlay,
                            cropped_image_np[:overlay_shape[0], :overlay_shape[1]],
                            original_image_np[y_offset:y_offset + overlay_shape[0], x_offset:x_offset + overlay_shape[1]]
                        )

                        mask_np[y_offset:y_offset + overlay_shape[0], x_offset:x_offset + overlay_shape[1]] = np.where(
                            cropped_mask_resized > 0,
                            cropped_mask_resized,
                            mask_np[y_offset:y_offset + overlay_shape[0], x_offset:x_offset + overlay_shape[1]]
                        )

                        objects_added += 1
                        break

                if objects_added < self.objects_to_add:
                    # print(f"Не удалось добавить {self.objects_to_add} объектов к изображению {os.path.basename(image_file)}. Добавлено только {objects_added}.")
                    pass

                # Применение аугментаций после добавления новых объектов
                if self.use_albumentations:
                    augmented = self.augmentation_pipeline(image=original_image_np, mask=mask_np)
                    original_image_np = augmented['image']
                    mask_np = augmented['mask']
                    original_image_np = np.clip(original_image_np, 0, 255).astype('uint8')
                    mask_np = np.clip(mask_np, 0, 255).astype('uint8')

                augmented_image_path = os.path.join(path, 'images', f'augmented_images_{img_dir}_{os.path.basename(image_file)}')
                augmented_mask_path = os.path.join(path, 'masks', f'augmented_images_{img_dir}_{os.path.basename(image_file).replace("rgb", "mask")}')

                Image.fromarray(original_image_np).save(augmented_image_path)
                Image.fromarray(mask_np).save(augmented_mask_path)
