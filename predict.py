# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import json
import glob
import time
import shutil
import zipfile
import tempfile
import pathlib
import subprocess
from typing import List, Tuple, Generator, Optional
from concurrent.futures import ThreadPoolExecutor
from random import shuffle

import numpy as np
import faiss
from cog import BasePredictor, Input, Path as CogPath

# Constants
WEIGHT_URLS = [
    # V1 pretrained models
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
    # V2 pretrained models
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
    # Hubert and RMVPE models
    (
        "https://weights.replicate.delivery/default/rvc/assets/hubert/hubert_base.pt",
        "assets/hubert/hubert_base.pt",
    ),
    (
        "https://weights.replicate.delivery/default/rvc/assets/rmvpe/rmvpe.pt",
        "assets/rmvpe/rmvpe.pt",
    ),
]

# Pretrained model paths by version and sample rate
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


class FileManager:
    """Manages file operations for the RVC model training pipeline."""

    @staticmethod
    def clean_workspace():
        """Clean up workspace by removing old files and directories."""
        print("Cleaning workspace...")

        os.makedirs("assets/weights", exist_ok=True)

        # Delete 'dataset' folder if it exists
        if os.path.exists("dataset"):
            shutil.rmtree("dataset")

        # Delete 'Model' folder if it exists
        if os.path.exists("Model"):
            shutil.rmtree("Model")

        # Delete contents of 'logs' folder but keep the folder and 'mute' directory
        if os.path.exists("logs"):
            for filename in os.listdir("logs"):
                file_path = os.path.join("logs", filename)
                if filename == "mute":
                    continue  # Skip the 'mute' directory
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    @staticmethod
    def extract_dataset(dataset_path: str):
        """Extract uploaded dataset zip file."""
        print(f"Extracting dataset from {dataset_path}...")
        with zipfile.ZipFile(dataset_path, "r") as zip_ref:
            zip_ref.extractall(".")

    @staticmethod
    def detect_model_name(base_path: str) -> Optional[str]:
        """Detect the model name from the dataset folder structure."""
        print(f"Detecting model name in {base_path}...")

        if not os.path.isdir(base_path):
            print(f"Directory does not exist: {base_path}")
            return None

        dirs = [
            d
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        ]
        return dirs[0] if dirs else None

    @staticmethod
    def prepare_directories(exp_dir: str):
        """Create necessary directories for the experiment."""
        os.makedirs(f"logs/{exp_dir}", exist_ok=True)
        with open(f"logs/{exp_dir}/preprocess.log", "w") as f:
            pass
        with open(f"logs/{exp_dir}/extract_f0_feature.log", "w") as f:
            pass

    @staticmethod
    def copy_model_files(exp_dir: str):
        """Copy necessary files to the model directory."""
        print("Copying model files...")
        os.makedirs(f"./Model/{exp_dir}", exist_ok=True)

        for file in glob.glob(f"logs/{exp_dir}/added_*.index"):
            print(f"Copying file: {file}")
            shutil.copy(file, f"./Model/{exp_dir}")

        for file in glob.glob(f"logs/{exp_dir}/total_*.npy"):
            print(f"Copying file: {file}")
            shutil.copy(file, f"./Model/{exp_dir}")

        weights_path = f"assets/weights/{exp_dir}.pth"
        if os.path.exists(weights_path):
            print(f"Copying file: {weights_path}")
            shutil.copy(weights_path, f"./Model/{exp_dir}")
        else:
            print(f"Warning: Weights file not found at {weights_path}")

    @staticmethod
    def create_archive(exp_dir: str) -> str:
        """Create a compressed archive of the model files."""
        print("Creating model archive...")
        base_dir = os.path.abspath(f"./Model/{exp_dir}")
        temp_7z_path = tempfile.mktemp(suffix=".7z")

        # Collect list of files to compress (relative to base_dir)
        files_to_add = [
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

        try:
            subprocess.run(
                ["7z", "a", "-t7z", "-mx=7", "-mmt=on", temp_7z_path] + files_to_add,
                cwd=base_dir,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            print(f"7z compression failed. STDOUT: {e.stdout.decode()}")
            print(f"STDERR: {e.stderr.decode()}")
            raise RuntimeError("7z archive creation failed") from e

        # Verify archive was created
        if not os.path.exists(temp_7z_path):
            raise RuntimeError("7z archive not created")

        return temp_7z_path


class WeightDownloader:
    """Handles downloading of weight files for the RVC model."""

    @staticmethod
    def download_weight(url: str, dest: str):
        """Download a single weight file."""
        # Create destination directory if it doesn't exist
        dest_dir = os.path.dirname(dest)
        os.makedirs(dest_dir, exist_ok=True)

        # Check if file already exists
        if not os.path.exists(dest):
            start = time.time()
            print(f"Downloading URL: {url}")
            print(f"Downloading to: {dest}")
            subprocess.check_call(["pget", url, dest], close_fds=False)
            print(f"Downloading took: {time.time() - start:.2f} seconds")
        else:
            print(f"File already exists: {dest}")

    @staticmethod
    def download_all_weights():
        """Download all required weight files in parallel."""
        print("Downloading model weights...")
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(WeightDownloader.download_weight, url, dest)
                for url, dest in WEIGHT_URLS
            ]

            # Wait for all downloads to complete
            for future in futures:
                future.result()


class TrainModules:
    """Handles the training modules and pipelines."""

    @staticmethod
    def run_command(command: str) -> Tuple[Optional[str], Optional[str]]:
        """Execute a shell command and return its output."""
        print(f"Running command: {command}")
        process = subprocess.Popen(command, shell=True)
        output, error = process.communicate()

        if process.returncode != 0:
            print(f"Error occurred: {error}")
        return output, error

    @staticmethod
    def process_data(dataset: str, sample_rate: str, exp_dir: str):
        """Process the dataset for training."""
        command = f"python infer/modules/train/preprocess.py '{dataset}' {sample_rate} 2 './logs/{exp_dir}' False 3.0"
        return TrainModules.run_command(command)

    @staticmethod
    def extract_features(f0method: str, exp_dir: str, version: str):
        """Extract f0 and other features from the processed data."""
        # Extract F0
        if f0method != "rmvpe_gpu":
            command = f"python infer/modules/train/extract/extract_f0_print.py './logs/{exp_dir}' 2 '{f0method}'"
        else:
            command = f"python infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 './logs/{exp_dir}' True"
        TrainModules.run_command(command)

        # Extract other features
        command = f"python infer/modules/train/extract_feature_print.py cuda:0 1 0 0 './logs/{exp_dir}' '{version}'"
        return TrainModules.run_command(command)

    @staticmethod
    def train_index(exp_dir: str, version: str) -> Generator[str, None, None]:
        """Train the index for feature matching."""
        exp_path = f"logs/{exp_dir}"
        os.makedirs(exp_path, exist_ok=True)

        feature_dir = (
            f"{exp_path}/3_feature256"
            if version == "v1"
            else f"{exp_path}/3_feature768"
        )
        if not os.path.exists(feature_dir):
            yield "Please perform feature extraction first!"
            return

        listdir_res = list(os.listdir(feature_dir))
        if len(listdir_res) == 0:
            yield "Please perform feature extraction first!"
            return

        # Load and concatenate features
        infos = []
        npys = []
        for name in sorted(listdir_res):
            phone = np.load(f"{feature_dir}/{name}")
            npys.append(phone)

        big_npy = np.concatenate(npys, 0)
        big_npy_idx = np.arange(big_npy.shape[0])
        np.random.shuffle(big_npy_idx)
        big_npy = big_npy[big_npy_idx]

        # Check if we need to do kmeans clustering
        if big_npy.shape[0] > 2e5:
            infos.append(
                f"Trying doing kmeans {big_npy.shape[0]} shape to 10k centers."
            )
            yield "\n".join(infos)
            try:
                from sklearn.cluster import MiniBatchKMeans
                import traceback

                big_npy = (
                    MiniBatchKMeans(
                        n_clusters=10000,
                        verbose=True,
                        batch_size=256 * 4,  # Using a default n_cpu value of 4
                        compute_labels=False,
                        init="random",
                    )
                    .fit(big_npy)
                    .cluster_centers_
                )
            except Exception as e:
                info = traceback.format_exc()
                infos.append(info)
                yield "\n".join(infos)

        # Save the features
        np.save(f"{exp_path}/total_fea.npy", big_npy)

        # Create and train the index
        feature_dim = 256 if version == "v1" else 768
        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        infos.append(f"{big_npy.shape}, {n_ivf}")
        yield "\n".join(infos)

        index = faiss.index_factory(feature_dim, f"IVF{n_ivf},Flat")
        infos.append("Training index...")
        yield "\n".join(infos)

        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        index.train(big_npy)

        # Save the trained index
        index_path = f"{exp_path}/trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir}_{version}.index"
        faiss.write_index(index, index_path)

        infos.append("Adding features to index...")
        yield "\n".join(infos)

        # Add features to the index in batches
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index.add(big_npy[i : i + batch_size_add])

        # Save the final index
        final_index_path = f"{exp_path}/added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir}_{version}.index"
        faiss.write_index(index, final_index_path)

        infos.append(
            f"Successfully built index: added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_dir}_{version}.index"
        )
        yield "\n".join(infos)

    @staticmethod
    def generate_filelist(
        exp_dir: str, sr: str, if_f0: bool, spk_id: int, version: str
    ) -> str:
        """Generate the filelist for training."""
        exp_path = f"logs/{exp_dir}"
        gt_wavs_dir = f"{exp_path}/0_gt_wavs"
        feature_dir = (
            f"{exp_path}/3_feature256"
            if version == "v1"
            else f"{exp_path}/3_feature768"
        )

        if if_f0:
            f0_dir = f"{exp_path}/2a_f0"
            f0nsf_dir = f"{exp_path}/2b-f0nsf"
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
        entries = []
        for name in names:
            if if_f0:
                # Fix: Use separate string formatting rather than backslashes in f-string expressions
                gt_path = gt_wavs_dir.replace("\\", "\\\\")
                feat_path = feature_dir.replace("\\", "\\\\")
                f0_path = f0_dir.replace("\\", "\\\\")
                f0nsf_path = f0nsf_dir.replace("\\", "\\\\")

                entries.append(
                    f"{gt_path}/{name}.wav|"
                    f"{feat_path}/{name}.npy|"
                    f"{f0_path}/{name}.wav.npy|"
                    f"{f0nsf_path}/{name}.wav.npy|"
                    f"{spk_id}"
                )
            else:
                gt_path = gt_wavs_dir.replace("\\", "\\\\")
                feat_path = feature_dir.replace("\\", "\\\\")

                entries.append(
                    f"{gt_path}/{name}.wav|" f"{feat_path}/{name}.npy|" f"{spk_id}"
                )

        # Add mute entries
        fea_dim = 256 if version == "v1" else 768
        if if_f0:
            for _ in range(2):
                entries.append(
                    f"./logs/mute/0_gt_wavs/mute{sr}.wav|"
                    f"./logs/mute/3_feature{fea_dim}/mute.npy|"
                    f"./logs/mute/2a_f0/mute.wav.npy|"
                    f"./logs/mute/2b-f0nsf/mute.wav.npy|"
                    f"{spk_id}"
                )
        else:
            for _ in range(2):
                entries.append(
                    f"./logs/mute/0_gt_wavs/mute{sr}.wav|"
                    f"./logs/mute/3_feature{fea_dim}/mute.npy|"
                    f"{spk_id}"
                )

        # Shuffle and write to file
        shuffle(entries)
        filelist_path = f"{exp_path}/filelist.txt"
        with open(filelist_path, "w") as f:
            f.write("\n".join(entries))

        return filelist_path

    @staticmethod
    def copy_config(exp_dir: str, sr: str, version: str):
        """Copy the configuration file to the experiment directory."""
        if version == "v1" or sr == "40k":
            config_path = f"configs/v1/{sr}.json"
        else:
            config_path = f"configs/v2/{sr}.json"

        config_save_path = f"logs/{exp_dir}/config.json"
        if not pathlib.Path(config_save_path).exists():
            with open(config_save_path, "w", encoding="utf-8") as f:
                with open(config_path, "r") as config_file:
                    config_data = json.load(config_file)
                    json.dump(
                        config_data,
                        f,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    )
                f.write("\n")

    @staticmethod
    def train_model(
        exp_dir: str,
        sr: str,
        if_f0: bool,
        spk_id: int,
        save_epoch: int,
        total_epoch: int,
        batch_size: str,
        if_save_latest: bool,
        pretrained_G: str,
        pretrained_D: str,
        gpus: str,
        if_cache_gpu: bool,
        if_save_every_weights: bool,
        version: str,
    ) -> str:
        """Train the RVC model."""
        # Generate filelist
        TrainModules.generate_filelist(exp_dir, sr, if_f0, spk_id, version)

        # Copy config
        TrainModules.copy_config(exp_dir, sr, version)

        # Build training command
        cmd = (
            f'python infer/modules/train/train.py -e "{exp_dir}" -sr {sr} -f0 {1 if if_f0 else 0} '
            f"-bs {batch_size} -g {gpus} -te {total_epoch} -se {save_epoch} "
            f'{"-pg %s" % pretrained_G if pretrained_G else ""} '
            f'{"-pd %s" % pretrained_D if pretrained_D else ""} '
            f"-l {1 if if_save_latest else 0} -c {1 if if_cache_gpu else 0} "
            f"-sw {1 if if_save_every_weights else 0} -v {version}"
        )

        # Run the training process
        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=".",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )

        # Print output in real time
        for line in process.stdout:
            print(line.strip())

        # Wait for process completion
        process.wait()

        return "Training completed. You can check the training log in the console."


class Predictor(BasePredictor):
    """Cog predictor for RVC model training."""

    def setup(self) -> None:
        """Set up the predictor by downloading required weights."""
        WeightDownloader.download_all_weights()

    def predict(
        self,
        dataset_zip: CogPath = Input(
            description="Upload dataset zip, zip should contain `dataset/<rvc_name>/split_<i>.wav`"
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
        try:
            # Clean workspace and extract dataset
            FileManager.clean_workspace()
            FileManager.extract_dataset(str(dataset_zip))

            # Detect model name from dataset directory
            model_name = FileManager.detect_model_name("dataset")
            if not model_name:
                raise ValueError("Could not detect model name from dataset directory")

            # Set up parameters
            exp_dir = model_name
            sample_rate_hz = "40000" if sample_rate == "40k" else "48000"
            dataset_path = f"dataset/{model_name}"

            # Create necessary directories
            FileManager.prepare_directories(exp_dir)

            # Process data
            print("Processing data...")
            TrainModules.process_data(dataset_path, sample_rate_hz, exp_dir)

            # Extract features
            print("Extracting features...")
            TrainModules.extract_features(f0method, exp_dir, version)

            # Train feature index
            print("Training feature index...")
            for progress in TrainModules.train_index(exp_dir, version):
                print(progress)

            # Get pretrained model paths
            G_path, D_path = PRETRAINED_PATHS[version][sample_rate]

            # Train model
            print("Training model...")
            training_result = TrainModules.train_model(
                exp_dir=exp_dir,
                sr=sample_rate,
                if_f0=True,
                spk_id=0,
                save_epoch=50,
                total_epoch=epoch,
                batch_size=batch_size,
                if_save_latest=True,
                pretrained_G=G_path,
                pretrained_D=D_path,
                gpus="0",
                if_cache_gpu=True,
                if_save_every_weights=False,
                version=version,
            )
            print(training_result)

            # Copy model files
            FileManager.copy_model_files(exp_dir)

            # Create archive
            archive_path = FileManager.create_archive(exp_dir)

            return CogPath(archive_path)

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise
