from datasets import load_dataset
import logging

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,  # Set the minimum level of messages to display
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define message format
)

# Create a logger for your module
logger = logging.getLogger(__name__)

logger.info("Started Dataset loading...")
data_files = {"train": "train.csv", "test": "test.csv"}
dataset = load_dataset(
    "imagefolder", 
    data_dir="dataset",
    num_proc=4
)
logger.info("Completed Dataset loading...")

logger.info("Started Dataset uploading...")
dataset.push_to_hub("chuuhtetnaing/myanmar-ocr-dataset", commit_message="created ocr data from 4 datasets")
logger.info("Completed Dataset uploading...")
