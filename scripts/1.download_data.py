import os
import urllib.request
import zipfile
from tqdm import tqdm
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Define the path for the 'data' folder one level up
data_folder_path = os.path.join('..', 'data')

# Check if the 'data' folder exists, if not, create it
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)
    print(f"{Fore.GREEN}Created 'data' folder at one level up.{Style.RESET_ALL}")
else:
    print(f"{Fore.YELLOW}'data' folder already exists at one level up.{Style.RESET_ALL}")

# Define the URL for the zip file and the local zip file path
zip_file_url = 'https://assets.01-edu.org/ai-branch/project5/home-credit-default-risk.zip'
local_zip_file_path = os.path.join(data_folder_path, 'home-credit-default-risk.zip')

# Function to show the progress bar
def download_with_progress(url, output_path):
    response = urllib.request.urlopen(url)
    total_size = int(response.info().get('Content-Length').strip())
    print(f"{Fore.CYAN}Total file size: {total_size / (1024 * 1024):.2f} MB{Style.RESET_ALL}")

    with open(output_path, 'wb') as out_file:
        chunk_size = 1024
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading", initial=0, ncols=100, ascii=False, bar_format='{l_bar}{bar}{r_bar}') as pbar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))
    response.close()

# Download the zip file with progress bar
try:
    print(f"{Fore.BLUE}Downloading the zip file...{Style.RESET_ALL}")
    download_with_progress(zip_file_url, local_zip_file_path)
    print(f"{Fore.GREEN}Downloaded the zip file.{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}Failed to download the zip file: {e}{Style.RESET_ALL}")
    exit(1)

# Unzip the file
try:
    print(f"{Fore.BLUE}Unzipping the file...{Style.RESET_ALL}")
    with zipfile.ZipFile(local_zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(data_folder_path)
    print(f"{Fore.GREEN}Unzipped the file.{Style.RESET_ALL}")
except zipfile.BadZipFile as e:
    print(f"{Fore.RED}Failed to unzip the file: {e}{Style.RESET_ALL}")
    exit(1)

print(f"{Fore.MAGENTA}Process completed.{Style.RESET_ALL}")