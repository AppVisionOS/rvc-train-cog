# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import glob
import json
import time
import string
import random
import pathlib
import tempfile
import requests
import shutil
import subprocess
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor
from subprocess import Popen, PIPE, STDOUT
from cog import BasePredictor, Input, Path as CogPath

# Pre-defined weight files to download
WEIGHT_FILES = [
    # Format: (url, destination)
    # Pretrained models (v1)
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/D32k.pth",
        "assets/pretrained/D32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/D40k.pth",
        "assets/pretrained/D40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/D48k.pth",
        "assets/pretrained/D48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/G32k.pth",
        "assets/pretrained/G32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/G40k.pth",
        "assets/pretrained/G40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/G48k.pth",
        "assets/pretrained/G48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0D32k.pth",
        "assets/pretrained/f0D32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0D40k.pth",
        "assets/pretrained/f0D40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0D48k.pth",
        "assets/pretrained/f0D48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0G32k.pth",
        "assets/pretrained/f0G32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0G40k.pth",
        "assets/pretrained/f0G40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained/f0G48k.pth",
        "assets/pretrained/f0G48k.pth",
    ),
    # Pretrained models (v2)
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/D32k.pth",
        "assets/pretrained_v2/D32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/D40k.pth",
        "assets/pretrained_v2/D40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/D48k.pth",
        "assets/pretrained_v2/D48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/G32k.pth",
        "assets/pretrained_v2/G32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/G40k.pth",
        "assets/pretrained_v2/G40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/G48k.pth",
        "assets/pretrained_v2/G48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0D32k.pth",
        "assets/pretrained_v2/f0D32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0D40k.pth",
        "assets/pretrained_v2/f0D40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0D48k.pth",
        "assets/pretrained_v2/f0D48k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0G32k.pth",
        "assets/pretrained_v2/f0G32k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0G40k.pth",
        "assets/pretrained_v2/f0G40k.pth",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/pretrained_v2/f0G48k.pth",
        "assets/pretrained_v2/f0G48k.pth",
    ),
    # Other models
    (
        "https://weights.replicate.delivery/default/rvc/assets/hubert/hubert_base.pt",
        "assets/hubert/hubert_base.pt",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/rmvpe/rmvpe.pt",
        "assets/rmvpe/rmvpe.pt",
    ),
]

# Pretrained model paths mapped by version and sample rate
PRETRAINED_PATHS = {
    "v1": {
        "40k": ("assets/pretrained/f0G40k.pth", "assets/pretrained/f0D40k.pth"),
        "48k": ("assets/pretrained/f0G48k.pth", "assets/pretrained/f0D48k.pth"),
    },
    "v2": {
        "40k": ("assets/pretrained_v2/f0G40k.pth", "assets/pretrained_v2/f0D40k.pth"),
        "48k": ("assets/pretrained_v2/f0G48k.pth", "assets/pretrained_v2/f0D48k.pth"),
    },
}


def download_file(url, dest):
    """Download a file from URL to destination if it doesn't exist."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if not os.path.exists(dest):
        start = time.time()
        print(f"Downloading {url} to {dest}")
        subprocess.check_call(["pget", url, dest], close_fds=False)
        print(f"Download completed in {time.time() - start:.2f}s")
    else:
        print(f"File already exists: {dest}")


def execute_command(command):
    """Execute a shell command and return its output."""
    process = subprocess.Popen(command, shell=True)
    output, error = process.communicate()

    if process.returncode != 0:
        print(f"Error occurred: {error}")

    return output, error


def get_first_directory(base_path):
    """Find the first directory in the specified path."""
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for directories in: {base_path}")

    if not os.path.isdir(base_path):
        print(f"Directory does not exist: {base_path}")
        return None

    dirs = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]
    return dirs[0] if dirs else None


def train_index(exp_dir1, version):
    """Train the feature index for the model."""

    exp_dir = f"logs/{exp_dir1}"
    os.makedirs(exp_dir, exist_ok=True)

    # Determine feature directory based on version
    feature_dim = "256" if version == "v1" else "768"
    feature_dir = f"{exp_dir}/3_feature{feature_dim}"

    if not os.path.exists(feature_dir):
        return "Please perform feature extraction first!"

    listdir_res = list(os.listdir(feature_dir))
    if not listdir_res:
        return "Please perform feature extraction first!"

    # Process features
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load(f"{feature_dir}/{name}")
        npys.append(phone)

    # Concatenate and shuffle features
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]

    # Apply k-means clustering for large feature arrays
    if big_npy.shape[0] > 2e5:
        import traceback
        from sklearn.cluster import MiniBatchKMeans

        infos.append(f"Trying doing kmeans {big_npy.shape[0]} shape to 10k centers.")
        yield "\n".join(infos)
        try:
            # Assuming n_cpu is a reasonable value like 4 if not defined
            n_cpu = getattr(config, "n_cpu", 4) if "config" in globals() else 4
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception as e:
            info = traceback.format_exc()
            print(f"Error in k-means clustering: {info}")
            infos.append(info)
            yield "\n".join(infos)

    # Save combined features
    np.save(f"{exp_dir}/total_fea.npy", big_npy)


def prepare_training_filelist(exp_dir1, sr2, if_f0_3, spk_id5, version19):
    """Prepare filelist for training."""
    exp_dir = f"./logs/{exp_dir1}"
    os.makedirs(exp_dir, exist_ok=True)

    # Define directories based on parameters
    gt_wavs_dir = f"{exp_dir}/0_gt_wavs"
    feature_dim = "256" if version19 == "v1" else "768"
    feature_dir = f"{exp_dir}/3_feature{feature_dim}"

    # Find common file names based on f0 settings
    if if_f0_3:
        f0_dir = f"{exp_dir}/2a_f0"
        f0nsf_dir = f"{exp_dir}/2b-f0nsf"
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )

    # Generate file entries
    file_entries = []
    for name in names:
        if if_f0_3:
            file_entries.append(
                f"{gt_wavs_dir.replace('\\', '\\\\')}\\{name}.wav|"
                f"{feature_dir.replace('\\', '\\\\')}\\{name}.npy|"
                f"{f0_dir.replace('\\', '\\\\')}\\{name}.wav.npy|"
                f"{f0nsf_dir.replace('\\', '\\\\')}\\{name}.wav.npy|{spk_id5}"
            )
        else:
            file_entries.append(
                f"{gt_wavs_dir.replace('\\', '\\\\')}\\{name}.wav|"
                f"{feature_dir.replace('\\', '\\\\')}\\{name}.npy|{spk_id5}"
            )

    # Add mute samples
    if if_f0_3:
        for _ in range(2):
            file_entries.append(
                f".\\logs\\mute\\0_gt_wavs\\mute{sr2}.wav|"
                f".\\logs\\mute\\3_feature{feature_dim}\\mute.npy|"
                f".\\logs\\mute\\2a_f0\\mute.wav.npy|"
                f".\\logs\\mute\\2b-f0nsf\\mute.wav.npy|{spk_id5}"
            )
    else:
        for _ in range(2):
            file_entries.append(
                f".\\logs\\mute\\0_gt_wavs\\mute{sr2}.wav|"
                f".\\logs\\mute\\3_feature{feature_dim}\\mute.npy|{spk_id5}"
            )

    # Shuffle and write filelist
    random.shuffle(file_entries)
    with open(f"{exp_dir}/filelist.txt", "w") as f:
        f.write("\n".join(file_entries))

    print("Filelist preparation complete")


def copy_config_file(exp_dir, sr, version):
    """Copy the appropriate config file for the model."""
    # Select config path based on version and sample rate
    if version == "v1" or sr == "40k":
        config_path = f"configs/v1/{sr}.json"
    else:
        config_path = f"configs/v2/{sr}.json"

    config_save_path = os.path.join(exp_dir, "config.json")

    # Copy config if it doesn't exist
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as out_file:
            with open(config_path, "r") as config_file:
                config_data = json.load(config_file)
                json.dump(
                    config_data,
                    out_file,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                )
            out_file.write("\n")


def run_training_process(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    """Run the model training process."""
    # Prepare filelist
    prepare_training_filelist(exp_dir1, sr2, if_f0_3, spk_id5, version19)

    # Setup config file
    exp_dir = f"./logs/{exp_dir1}"
    copy_config_file(exp_dir, sr2, version19)

    # Log training parameters
    print(f"Using GPUs: {gpus16}")
    if not pretrained_G14:
        print("No pretrained Generator")
    if not pretrained_D15:
        print("No pretrained Discriminator")

    # Build training command
    cmd = (
        f'python infer/modules/train/train.py -e "{exp_dir1}" '
        f"-sr {sr2} -f0 {1 if if_f0_3 else 0} -bs {batch_size12} "
        f"-g {gpus16} -te {total_epoch11} -se {save_epoch10} "
        f'{"-pg " + pretrained_G14 if pretrained_G14 else ""} '
        f'{"-pd " + pretrained_D15 if pretrained_D15 else ""} '
        f"-l {1 if if_save_latest13 else 0} "
        f"-c {1 if if_cache_gpu17 else 0} "
        f"-sw {1 if if_save_every_weights18 else 0} "
        f"-v {version19}"
    )

    # Run training process
    p = Popen(
        cmd,
        shell=True,
        cwd=".",
        stdout=PIPE,
        stderr=STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    # Stream output
    for line in p.stdout:
        print(line.strip())

    p.wait()
    return "Training completed. You can check the training log in the console or the 'train.log' file."


def create_model_archive(exp_dir):
    """Create a compressed archive of the model files."""
    # Create model directory
    model_dir = f"./Model/{exp_dir}"
    os.makedirs(model_dir, exist_ok=True)

    # Copy model files
    print("Copying model files...")

    # Copy index files
    for file in glob.glob(f"logs/{exp_dir}/added_*.index"):
        print(f"Copying: {file}")
        shutil.copy(file, model_dir)

    # Copy feature files
    for file in glob.glob(f"logs/{exp_dir}/total_*.npy"):
        print(f"Copying: {file}")
        shutil.copy(file, model_dir)

    # Copy weights
    weights_file = f"assets/weights/{exp_dir}.pth"
    print(f"Copying: {weights_file}")
    shutil.copy(weights_file, model_dir)

    # Create archive
    base_dir = os.path.abspath(model_dir)
    temp_archive_path = tempfile.mktemp(suffix=".7z")

    # Collect files to archive (relative paths)
    files_to_archive = [
        f"{exp_dir}.pth",
        *[
            os.path.relpath(f, base_dir)
            for f in glob.glob(os.path.join(base_dir, "added_*.index"))
        ],
        *[
            os.path.relpath(f, base_dir)
            for f in glob.glob(os.path.join(base_dir, "total_*.npy"))
        ],
    ]

    # Create archive
    try:
        subprocess.run(
            ["7z", "a", "-t7z", "-mx=7", "-mmt=on", temp_archive_path]
            + files_to_archive,
            cwd=base_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        print(f"Archive creation failed. STDOUT: {e.stdout.decode()}")
        print(f"STDERR: {e.stderr.decode()}")
        raise RuntimeError("Failed to create model archive") from e

    # Verify archive exists
    if not os.path.exists(temp_archive_path):
        raise RuntimeError("Archive creation failed: file not found")

    return temp_archive_path


class Predictor(BasePredictor):
    def setup(self):
        """Set up the predictor by downloading required model weights."""
        # Download weights in parallel
        with ThreadPoolExecutor() as executor:
            executor.map(lambda args: download_file(*args), WEIGHT_FILES)

    def cleanup_workspace(self):
        """Clean up workspace before starting a new prediction."""
        # Create weights directory
        os.makedirs("assets/weights", exist_ok=True)

        # Remove temporary directories
        for dir_path in ["dataset", "Model"]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

        # Clean logs directory but keep 'mute'
        if os.path.exists("logs"):
            for item in os.listdir("logs"):
                item_path = os.path.join("logs", item)
                if item == "mute":
                    continue  # Preserve mute directory
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

    def predict(
        self,
        wav_urls: list[str] = Input(
            description="Array of WAV file URLs to use as dataset samples"
        ),
        sample_rate: str = Input(
            description="Sample rate", default="48k", choices=["40k", "48k"]
        ),
        version: str = Input(description="Version", default="v2", choices=["v1", "v2"]),
        f0method: str = Input(
            description="F0 method, `rmvpe_gpu` recommended.",
            default="rmvpe_gpu",
            choices=["pm", "dio", "harvest", "rmvpe", "rmvpe_gpu"],
        ),
        epoch: int = Input(description="Epoch", default=10),
        batch_size: str = Input(description="Batch size", default="7"),
    ) -> CogPath:
        """Train RVC model with provided audio samples."""
        # Clean workspace
        self.cleanup_workspace()

        # Generate model name and create dataset directory
        timestamp = int(time.time())
        random_str = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=6)
        )
        model_name = f"rvc_{timestamp}_{random_str}"
        dataset_path = f"dataset/{model_name}"
        os.makedirs(dataset_path, exist_ok=True)

        # Download audio samples
        for i, url in enumerate(wav_urls):
            output_path = f"{dataset_path}/split_{i}.wav"
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {url} to {output_path}")
            except Exception as e:
                print(f"Failed to download {url}: {str(e)}")

        # Get model name from dataset directory
        model_name = get_first_directory("dataset")

        # Set up parameters
        num_samples = "40000" if sample_rate == "40k" else "48000"
        k_sample_rate = "40k" if sample_rate == "40k" else "48k"
        dataset_dir = f"dataset/{model_name}"
        exp_dir = model_name
        save_frequency = 50
        cache_gpu = True

        # Create log directories
        os.makedirs(f"./logs/{exp_dir}", exist_ok=True)
        open(f"./logs/{exp_dir}/preprocess.log", "w").close()
        open(f"./logs/{exp_dir}/extract_f0_feature.log", "w").close()

        # Preprocess data
        print("Preprocessing data...")
        execute_command(
            f"python infer/modules/train/preprocess.py '{dataset_dir}' {num_samples} "
            f"2 './logs/{exp_dir}' False 3.0"
        )

        # Extract F0
        print("Extracting F0...")
        if f0method != "rmvpe_gpu":
            execute_command(
                f"python infer/modules/train/extract/extract_f0_print.py './logs/{exp_dir}' "
                f"2 '{f0method}'"
            )
        else:
            execute_command(
                f"python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 './logs/{exp_dir}' True"
            )

        # Extract features
        print("Extracting features...")
        execute_command(
            f"python infer/modules/train/extract_feature_print.py cuda:0 1 0 0 './logs/{exp_dir}' '{version}'"
        )

        # Train feature index
        print("Training feature index...")
        for result in train_index(exp_dir, version):
            print(result)

        # Train model
        print("Training model...")
        G_path, D_path = PRETRAINED_PATHS[version][k_sample_rate]
        run_training_process(
            exp_dir,
            k_sample_rate,
            True,
            0,
            save_frequency,
            epoch,
            batch_size,
            True,
            G_path,
            D_path,
            0,
            cache_gpu,
            False,
            version,
        )

        # Create model archive
        print("Creating model archive...")
        archive_path = create_model_archive(exp_dir)

        return CogPath(archive_path)
