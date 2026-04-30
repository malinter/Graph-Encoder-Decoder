import os

def list_registry():
    # Use environment variable for the registry path, defaulting to a relative path
    registry_path = os.environ.get('MAO_REGISTRY_PATH', os.path.join(os.getcwd(), 'registry'))
    print(f"Checking access to: {registry_path}")
    
    if not os.path.exists(registry_path):
        print(f"Error: The directory {registry_path} does not exist.")
        return

    try:
        files = os.listdir(registry_path)
        print(f"Successfully accessed registry. Found {len(files)} items:")
        for file in files:
            print(f" - {file}")
    except PermissionError:
        print(f"Error: Permission denied (Operation not permitted).")
        print("Tip: Ensure the terminal/agent has 'Full Disk Access' or 'Files and Folders' permissions in macOS System Settings.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    list_registry()
