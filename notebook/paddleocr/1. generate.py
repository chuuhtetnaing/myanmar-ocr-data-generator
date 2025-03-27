import random
import multiprocessing as mp
import os
import uuid
from functools import partial
import sys
sys.path.append('../../')

from trdg.generators import GeneratorFromStrings
import logging

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,  # Set the minimum level of messages to display
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define message format
)

# Create a logger for your module
logger = logging.getLogger(__name__)


root_dir = "dataset"


def process_string(output_dir, dataset_type, param_dict, string):
    # Create a generator for a single string
    generator = GeneratorFromStrings(
        strings=[string],  # Wrap the string in a list
        language='my',
        count=1,  # Generate one image for this string
        **param_dict
    )

    # Get the generated item
    try:
        item = next(generator)
        img, lbl = item

        # Skip if image is None
        if img is None:
            return None, lbl

        # Create a unique filename
        img_filename = f"{str(uuid.uuid4())}.jpg"
        img_path = os.path.join(output_dir, img_filename)

        # Save the image
        img.save(img_path)

        # Write annotation to file (in a thread-safe way)
        annotations_file = os.path.join(root_dir, f"{dataset_type}.txt")
        with open(annotations_file, 'a') as f:
            f.write(f"{dataset_type}/{img_filename}\t{lbl}\n")

        return img_path, lbl
    except Exception as e:
        print(f"Error processing string '{string}': {str(e)}")
        return None, string


def generate_images_parallel(corpus, output_dir, dataset_type, image_params, num_processes=None):
    # Create a pool of workers
    pool = mp.Pool(processes=num_processes or mp.cpu_count())

    # Create a partial function with fixed arguments
    process_func = partial(process_string, output_dir, dataset_type, image_params)

    # Process strings in parallel
    results = pool.map(process_func, corpus)

    # Close the pool
    pool.close()
    pool.join()

    # Count successful generations
    successful = sum(1 for r in results if r[0] is not None)
    print(f"Generated {successful} images with annotations in {dataset_type}")


def shuffle_and_split(data, train_ratio=0.9):
    # Create a copy of the list to avoid modifying the original
    shuffled_data = data.copy()

    # Shuffle the list in place
    random.shuffle(shuffled_data)

    # Calculate the split point
    split_point = int(len(shuffled_data) * train_ratio)

    # Split the list into training and test sets
    train_set = shuffled_data[:split_point]
    test_set = shuffled_data[split_point:]

    return train_set, test_set


if __name__ == "__main__":
    
    logger.info("Split into Train and Test Dataset...")
    with open("data/my_corpus.txt") as file:
        my_corpus = file.read().split("\n")
    
    my_corpus_train, my_corpus_test = shuffle_and_split(my_corpus)
    
    with open("data/my_corpus_train.txt", "w") as file:
        file.write("\n".join(my_corpus_train))

    with open("data/my_corpus_test.txt", "w") as file:
        file.write("\n".join(my_corpus_test))
    
    
    
    # Generate Train Images
    logger.info("Generate Train Images...")
    dataset_type = "train"
    output_dir = root_dir + f"/{dataset_type}"
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Path for annotations file
    annotations_file = os.path.join(root_dir, f"{dataset_type}.txt")

    # Clear annotations file if it exists
    with open(annotations_file, 'w') as f:
        pass

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    ## Simple Images
    logger.info("Generate Simple Images...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }
    generate_images_parallel(my_corpus, output_dir, dataset_type, image_params)
    
    ## Skew Images
    logger.info("Generate Skew Images...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:100_000], output_dir, dataset_type, image_params)
    

    
    ## Distorsion Images
    
    ### 1. (distorsion_type: 1, distorsion_orientation: 0)
    logger.info("Generate Distorsion Images 1...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 1,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:10_000], output_dir, dataset_type, image_params)
    
    ### 2. (distorsion_type: 1, distorsion_orientation: 1)
    logger.info("Generate Distorsion Images 2...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 1,
        "distorsion_orientation": 1,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:10_000], output_dir, dataset_type, image_params)
    
    ### 3. (distorsion_type: 1, distorsion_orientation: 2)
    logger.info("Generate Distorsion Images 3...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 1,
        "distorsion_orientation": 2,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:10_000], output_dir, dataset_type, image_params)
    
    ### 4. (distorsion_type: 2, distorsion_orientation: 0)
    logger.info("Generate Distorsion Images 4...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 2,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:10_000], output_dir, dataset_type, image_params)
    
    ### 5. (distorsion_type: 2, distorsion_orientation: 1)
    logger.info("Generate Distorsion Images 5...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 2,
        "distorsion_orientation": 1,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:10_000], output_dir, dataset_type, image_params)
    
    ### 6. (distorsion_type: 2, distorsion_orientation: 2)
    logger.info("Generate Distorsion Images 6...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 2,
        "distorsion_orientation": 2,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:10_000], output_dir, dataset_type, image_params)
    
    ### 7. (distorsion_type: 3, distorsion_orientation: 0)
    logger.info("Generate Distorsion Images 7...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 3,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:10_000], output_dir, dataset_type, image_params)
    
    ### 8. (distorsion_type: 3, distorsion_orientation: 1)
    logger.info("Generate Distorsion Images 8...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 3,
        "distorsion_orientation": 1,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:10_000], output_dir, dataset_type, image_params)
    
    ### 9. (distorsion_type: 3, distorsion_orientation: 2)
    logger.info("Generate Distorsion Images 9...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 3,
        "distorsion_orientation": 2,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:10_000], output_dir, dataset_type, image_params)
    

    
    ## Blur Images
    
    ### 1. (blur: 0)
    logger.info("Generate Blur Images 1...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:25_000], output_dir, dataset_type, image_params)
    

    
    ### 2. (blur: 1)
    logger.info("Generate Blur Images 2...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 1,
        "random_blur": False,
        "background_type": 0
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:25_000], output_dir, dataset_type, image_params)
    
    ### 3. (blur: 2)
    logger.info("Generate Blur Images 3...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 2,
        "random_blur": False,
        "background_type": 0
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:25_000], output_dir, dataset_type, image_params)
    
    ### 4. (blur: 4)
    logger.info("Generate Blur Images 4...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 4,
        "random_blur": False,
        "background_type": 0
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:25_000], output_dir, dataset_type, image_params)
    

    

    
    ## Background Images
    
    ### 1. (background_type: 0)
    logger.info("Generate Background Images 1...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:25_000], output_dir, dataset_type, image_params)
    

    
    ### 2. (background_type: 1)
    logger.info("Generate Background Images 2...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 1
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:25_000], output_dir, dataset_type, image_params)
    
    ### 3. (background_type: 2)
    logger.info("Generate Background Images 3...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 2
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:25_000], output_dir, dataset_type, image_params)
    
    ### 4. (background_type: 3)
    logger.info("Generate Background Images 4...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 3
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:25_000], output_dir, dataset_type, image_params)
    

    

    
    # Generate Test Images
    logger.info("Generate Train Images...")
    dataset_type = "test"
    output_dir = root_dir + f"/{dataset_type}"
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Path for annotations file
    annotations_file = os.path.join(root_dir, f"{dataset_type}.txt")

    # Clear annotations file if it exists
    with open(annotations_file, 'w') as f:
        pass

    with open("data/my_corpus_test.txt") as file:
        my_corpus = file.read().split("\n")
    
    ## Simple Images
    logger.info("Generate Simple Images...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    generate_images_parallel(my_corpus, output_dir, dataset_type, image_params)
    
    ## Skew Images
    logger.info("Generate Skew Images...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:10_000], output_dir, dataset_type, image_params)
    

    
    ## Distorsion Images
    
    ### 1. (distorsion_type: 1, distorsion_orientation: 0)
    logger.info("Generate Distorsion Images 1...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 1,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:1_000], output_dir, dataset_type, image_params)
    
    ### 2. (distorsion_type: 1, distorsion_orientation: 1)
    logger.info("Generate Distorsion Images 2...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 1,
        "distorsion_orientation": 1,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:1_000], output_dir, dataset_type, image_params)
    
    ### 3. (distorsion_type: 1, distorsion_orientation: 2)
    logger.info("Generate Distorsion Images 3...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 1,
        "distorsion_orientation": 2,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:1_000], output_dir, dataset_type, image_params)
    
    ### 4. (distorsion_type: 2, distorsion_orientation: 0)
    logger.info("Generate Distorsion Images 4...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 2,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:1_000], output_dir, dataset_type, image_params)
    
    ### 5. (distorsion_type: 2, distorsion_orientation: 1)
    logger.info("Generate Distorsion Images 5...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 2,
        "distorsion_orientation": 1,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:1_000], output_dir, dataset_type, image_params)
    
    ### 6. (distorsion_type: 2, distorsion_orientation: 2)
    logger.info("Generate Distorsion Images 6...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 2,
        "distorsion_orientation": 2,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:1_000], output_dir, dataset_type, image_params)
    
    ### 7. (distorsion_type: 3, distorsion_orientation: 0)
    logger.info("Generate Distorsion Images 7...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 3,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:1_000], output_dir, dataset_type, image_params)
    
    ### 8. (distorsion_type: 3, distorsion_orientation: 1)
    logger.info("Generate Distorsion Images 8...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 3,
        "distorsion_orientation": 1,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:1_000], output_dir, dataset_type, image_params)
    
    ### 9. (distorsion_type: 3, distorsion_orientation: 2)
    logger.info("Generate Distorsion Images 9...")
    image_params = {
        "skewing_angle": 5,
        "random_skew": True,
        "distorsion_type": 3,
        "distorsion_orientation": 2,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:1_000], output_dir, dataset_type, image_params)
    

    
    ## Blur Images
    
    ### 1. (blur: 0)
    logger.info("Generate Blur Images 1...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:2_500], output_dir, dataset_type, image_params)
    

    
    ### 2. (blur: 1)
    logger.info("Generate Blur Images 2...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 1,
        "random_blur": False,
        "background_type": 0
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:2_500], output_dir, dataset_type, image_params)
    
    ### 3. (blur: 2)
    logger.info("Generate Blur Images 3...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 2,
        "random_blur": False,
        "background_type": 0
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:2_500], output_dir, dataset_type, image_params)
    
    ### 4. (blur: 4)
    logger.info("Generate Blur Images 4...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 4,
        "random_blur": False,
        "background_type": 0
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:2_500], output_dir, dataset_type, image_params)
    

    

    
    ## Background Images
    
    ### 1. (background_type: 0)
    logger.info("Generate Background Images 1...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 0
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:2_500], output_dir, dataset_type, image_params)
    

    
    ### 2. (background_type: 1)
    logger.info("Generate Background Images 2...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 1
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:2_500], output_dir, dataset_type, image_params)
    
    ### 3. (background_type: 2)
    logger.info("Generate Background Images 3...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 2
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:2_500], output_dir, dataset_type, image_params)
    
    ### 4. (background_type: 3)
    logger.info("Generate Background Images 4...")
    image_params = {
        "skewing_angle": 0,
        "random_skew": False,
        "distorsion_type": 0,
        "distorsion_orientation": 0,
        "blur": 0,
        "random_blur": False,
        "background_type": 3
    }

    with open("data/my_corpus_train.txt") as file:
        my_corpus = file.read().split("\n")

    random.shuffle(my_corpus)

    generate_images_parallel(my_corpus[:2_500], output_dir, dataset_type, image_params)