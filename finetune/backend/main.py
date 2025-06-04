import paramiko
from scp import SCPClient
import os
import logging
from config import *  # Import all configuration variables

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# -------------------------------------------------------
# -------------------------------------------------------
# Create SSH client
try:
    logger.info(f"Connecting to VM at {VM_IP}")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(VM_IP, username=USERNAME, password=VM_PASSWORD)
    logger.info("Successfully connected to VM")
except Exception as e:
    logger.error(f"Failed to connect to VM: {str(e)}")
    raise

# Update file paths to use relative paths from the backend directory
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.py")
FINETUNE_PATH = os.path.join(os.path.dirname(__file__), "finetune.py")
DATASET_PATH = os.path.join(os.path.dirname(__file__), DATASET_PATH)  # Use dataset path from config

# Upload config.py first
try:
    logger.info("Starting file upload: config.py")
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(CONFIG_PATH, "~/config.py")
    logger.info("Successfully uploaded config.py")
except Exception as e:
    logger.error(f"Failed to upload config file: {str(e)}")
    raise

# Upload dataset if FLAG_UPLOAD_DATASET is True
if FLAG_UPLOAD_DATASET:
    try:
        logger.info("Starting dataset upload")
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(DATASET_PATH, "~/dataset", recursive=True)
        logger.info("Successfully uploaded dataset")
    except Exception as e:
        logger.error(f"Failed to upload dataset: {str(e)}")
        raise

# Upload finetune.py
try:
    logger.info("Starting file upload: finetune.py")
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(FINETUNE_PATH, "~/finetune.py")
    logger.info("Successfully uploaded finetune.py")
except Exception as e:
    logger.error(f"Failed to upload file: {str(e)}")
    raise

# Upload dataset if flag is set
if FLAG_UPLOAD_DATASET:
    try:
        logger.info(f"Starting dataset upload from: {DATASET_PATH}")
        
        # Create the dataset directory on VM if it doesn't exist
        stdin, stdout, stderr = ssh.exec_command("mkdir -p ~/dataset")
        if stdout.channel.recv_exit_status() != 0:
            raise Exception("Failed to create dataset directory on VM")
            
        # Upload the dataset
        with SCPClient(ssh.get_transport()) as scp:
            # If DATASET_PATH points to a directory, upload its contents
            if os.path.isdir(DATASET_PATH):
                for root, dirs, files in os.walk(DATASET_PATH):
                    for file in files:
                        local_path = os.path.join(root, file)
                        # Calculate relative path to maintain directory structure
                        rel_path = os.path.relpath(local_path, DATASET_PATH)
                        remote_path = f"~/dataset/{rel_path}"
                        # Create remote directory if needed
                        remote_dir = os.path.dirname(remote_path)
                        ssh.exec_command(f"mkdir -p {remote_dir}")
                        # Upload file
                        scp.put(local_path, remote_path)
                        logger.info(f"Uploaded: {rel_path}")
            # If DATASET_PATH points to a single file
            else:
                scp.put(DATASET_PATH, "~/dataset/")
                logger.info(f"Uploaded single file: {os.path.basename(DATASET_PATH)}")
                
        logger.info("Successfully uploaded dataset")
    except Exception as e:
        logger.error(f"Failed to upload dataset: {str(e)}")
        raise

# Add .env file upload before running finetune.py
try:
    logger.info("Starting file upload: .env")
    ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(ENV_PATH, "~/.env")
    logger.info("Successfully uploaded .env")
except Exception as e:
    logger.error(f"Failed to upload .env file: {str(e)}")
    raise

# Run commands on the VM
if FLAG_CREATE_ENV:
    base_commands = [
        # Combine commands with proper shell initialization and full conda path
        """bash -l -c 'eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
        conda activate && \
        pip3 install --upgrade pip && \
        pip3 install jinja2>=3.1.0 && \
        pip3 install Pillow==10.1.0 && \
        pip3 install transformers==4.46.2 && \
        pip3 install trl==0.13.0 && \
        pip3 install unsloth && \
        pip3 install python-dotenv && \
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
        python3 finetune.py'"""
    ]
else:
    base_commands = [
        """bash -l -c 'eval "$(~/miniconda3/bin/conda shell.bash hook)" && \
        conda activate && \
        python3 finetune.py'"""
    ]

# Only add compression commands if testing locally
if FLAG_TESTINLOCAL:
    base_commands.extend([
        """bash -l -c 'cd {} && \
           tar -czvf {}.tar.gz adapter_model.safetensors adapter_config.json tokenizer.json tokenizer_config.json chat_template.json preprocessor_config.json README.md special_tokens_map.json'""".format(MODEL_NAME, MODEL_NAME)
    ])

for command in base_commands:
    try:
        logger.info(f"Executing command: {command}")
        stdin, stdout, stderr = ssh.exec_command(command)
        
        # Log stdout in real-time
        for line in stdout:
            logger.info(f"STDOUT: {line.strip()}")
        
        # Log stderr in real-time
        for line in stderr:
            logger.error(f"STDERR: {line.strip()}")
        
        # Check command exit status
        if stdout.channel.recv_exit_status() != 0:
            logger.error(f"Command failed with non-zero exit status: {command}")
            raise Exception(f"Command failed: {command}")
        
        logger.info(f"Successfully executed command: {command}")
    except Exception as e:
        logger.error(f"Failed to execute command '{command}': {str(e)}")
        raise

# Only download if testing locally
if FLAG_TESTINLOCAL:
    try:
        logger.info(f"Starting download of {MODEL_NAME}.tar.gz")
        with SCPClient(ssh.get_transport()) as scp:
            scp.get(f"~/{MODEL_NAME}/{MODEL_NAME}.tar.gz", LOCAL_PATH)
        logger.info(f"Successfully downloaded {MODEL_NAME}.tar.gz to {LOCAL_PATH}")
        # Install Ollama on the remote VM
        try:
            logger.info("Installing Ollama on remote VM")
            ollama_install_commands = [
                # Download and install Ollama
                """bash -l -c 'curl -fsSL https://ollama.com/install.sh | sh'"""
            ]
            
            for command in ollama_install_commands:
                logger.info(f"Executing command: {command}")
                stdin, stdout, stderr = ssh.exec_command(command)
                
                # Log stdout in real-time
                for line in stdout:
                    logger.info(f"STDOUT: {line.strip()}")
                
                # Log stderr in real-time  
                for line in stderr:
                    logger.error(f"STDERR: {line.strip()}")
                
                # Check command exit status
                if stdout.channel.recv_exit_status() != 0:
                    logger.error(f"Ollama installation command failed with non-zero exit status: {command}")
                    raise Exception(f"Ollama installation failed: {command}")
                
                logger.info("Successfully installed Ollama")
                    
        except Exception as e:
            logger.error(f"Failed to install Ollama: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Failed to download model file: {str(e)}")
        raise

logger.info("Closing SSH connection")
ssh.close()
logger.info("Script completed successfully")
