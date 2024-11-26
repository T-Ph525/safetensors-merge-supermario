import os
import shutil
import argparse
import requests
from tqdm import tqdm
from huggingface_hub import HfApi, Repository, hf_hub_download, upload_folder
from merge import merge_folder, map_tensors_to_files, copy_nontensor_files, save_tensor_map
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class RepositoryManager:
    """
    A class to manage HuggingFace repositories.
    """
    base_model_path = os.path.join(os.getcwd(), "base_model")

    def __init__(self, repo_id=None, token=None):
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token) if token else HfApi()

    def download_repo(self, repo_name, path):
        """
        Download a repository from HuggingFace.

        Args:
            repo_name (str): The name of the repository.
            path (str): The path to save the downloaded repository.
        """
        if os.path.isdir(repo_name):
            if not os.path.exists(path):
                os.makedirs(path)
            shutil.copytree(repo_name, path, dirs_exist_ok=True)
        else:
            if not os.path.exists(path):
                os.makedirs(path)

            repo_files = self.api.list_repo_files(repo_name)

            for file_path in tqdm(repo_files, desc=f"Downloading {repo_name}"):
                file_url = f"https://huggingface.co/{repo_name}/resolve/main/{file_path}"
                hf_hub_download(repo_id=repo_name, filename=file_path, cache_dir=path, local_dir=path)

    def delete_repo(self, path):
        """
        Delete a repository from the local filesystem.

        Args:
            path (str): The path to the repository.
        """
        shutil.rmtree(path, ignore_errors=True)

class ModelMerger:
    """
    A class to merge models and upload them to HuggingFace.
    """
    def __init__(self, repo_id=None, token=None):
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token) if token else HfApi()
        self.tensor_map = None

    def prepare_base_model(self, base_model_name, base_model_path):
        """
        Prepare the base model by downloading it from HuggingFace.

        Args:
            base_model_name (str): The name of the base model.
            base_model_path (str): The path to save the base model.
        """
        repo_manager = RepositoryManager(self.repo_id, self.token)
        repo_manager.download_repo(base_model_name, base_model_path)
        self.tensor_map = map_tensors_to_files(base_model_path)

    def merge_repo(self, repo_name, repo_path, p, lambda_val):
        """
        Merge the base model with another model from HuggingFace.

        Args:
            repo_name (str): The name of the model to merge.
            repo_path (str): The path to save the model to merge.
            p (float): Dropout probability.
            lambda_val (float): Scaling factor.
        """
        repo_manager = RepositoryManager(self.repo_id, self.token)
        repo_manager.delete_repo(repo_path)
        repo_manager.download_repo(repo_name, repo_path)

        try:
            self.tensor_map = merge_folder(self.tensor_map, repo_path, p, lambda_val)
            logging.info(f"Merged {repo_name}")
        except Exception as e:
            logging.error(f"Error merging {repo_name}: {e}")

    def finalize_merge(self, output_dir):
        """
        Finalize the merge by copying non-tensor files and saving the merged tensor map.

        Args:
            output_dir (str): The path to the output directory.
        """
        base_model_path = os.path.join(os.getcwd(), "base_model")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        copy_nontensor_files(base_model_path, output_dir)
        save_tensor_map(self.tensor_map, output_dir)

    def upload_model(self, output_dir, repo_name, commit_message):
        """
        Upload the merged model to HuggingFace.

        Args:
            output_dir (str): The path to the output directory.
            repo_name (str): The name of the repository to upload to.
            commit_message (str): The commit message for the upload.
        """
        repo = Repository(local_dir=output_dir, clone_from=repo_name, use_auth_token=self.token)
        repo.git_add(auto_lfs_track=True)
        repo.git_commit(commit_message)
        repo.git_push()
        logging.info(f"Model uploaded to {repo_name}")

def get_max_vocab_size(repo_list):
    """
    Get the maximum vocabulary size from a list of repositories.

    Args:
        repo_list (list): A list of repositories.

    Returns:
        tuple: A tuple containing the maximum vocabulary size and the repository with the maximum vocabulary size.
    """
    max_vocab_size = 0
    repo_with_max_vocab = None
    base_url = "https://huggingface.co/{}/raw/main/config.json"

    for repo_name, _, _ in repo_list:
        url = base_url.format(repo_name)
        try:
            response = requests.get(url)
            config = response.json()
            vocab_size = config.get('vocab_size', 0)
            if vocab_size > max_vocab_size:
                max_vocab_size = vocab_size
                repo_with_max_vocab = repo_name
        except requests.RequestException as e:
            logging.error(f"Error fetching vocab size from {repo_name}: {e}")

    return max_vocab_size, repo_with_max_vocab

def download_json_files(repo_name, file_paths, output_dir):
    """
    Download JSON files from a repository.

    Args:
        repo_name (str): The name of the repository.
        file_paths (list): A list of file paths to download.
        output_dir (str): The path to save the downloaded files.
    """
    base_url = f"https://huggingface.co/{repo_name}/raw/main/"
    for file_path in file_paths:
        url = base_url + file_path
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join(output_dir, os.path.basename(file_path)), 'wb') as file:
                file.write(response.content)
        else:
            logging.error(f"Failed to download {file_path} from {repo_name}")

def main():
    """
    Main function to parse command-line arguments and orchestrate the merging and uploading process.
    """
    parser = argparse.ArgumentParser(description="Merge and upload HuggingFace models")
    parser.add_argument('base_model', type=str, help='Base model safetensors file')
    parser.add_argument('model_to_merge', type=str, help='Model to merge (.safetensors or .bin)')
    parser.add_argument('-p', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('-lambda', '--lambda_value', type=float, default=3.0, help='Scaling factor (optional)')
    parser.add_argument('--token', type=str, help='HuggingFace token (required for uploading)')
    parser.add_argument('--repo', type=str, help='HuggingFace repo to upload to (required for uploading)')
    parser.add_argument('--commit-message', type=str, default='Upload merged model', help='Commit message for model upload')
    parser.add_argument('-U', '--upload', action='store_true', help='Upload the merged model to HuggingFace Hub')
    args = parser.parse_args()

    base_model_path = os.path.join(os.getcwd(), "base_model")
    model_to_merge_path = os.path.join(os.getcwd(), "model_to_merge")
    output_dir = os.path.join(os.getcwd(), "output")

    model_merger = ModelMerger(args.repo, args.token)
    model_merger.prepare_base_model(args.base_model, base_model_path)

    model_merger.merge_repo(args.model_to_merge, model_to_merge_path, args.p, args.lambda_value)

    model_merger.finalize_merge(output_dir)

    if args.upload:
        if not args.token or not args.repo:
            logging.error("Error: HuggingFace token and repo name are required for uploading.")
        else:
            model_merger.upload_model(output_dir, args.repo, args.commit_message)

if __name__ == "__main__":
    main()
