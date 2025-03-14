import cv2
import os
import shutil
from pathlib import Path

def extract_frames_and_manage_folders(video_path, original_img1_path, temp_storage_path):
    """
    Extract frames from video and manage folder swapping
    
    Args:
        video_path (str): Path to the input video file
        original_img1_path (str): Path to the original img1 folder
        temp_storage_path (str): Path where to temporarily store the original img1 folder
    """
    # Create necessary directories if they don't exist
    os.makedirs('img1', exist_ok=True)
    os.makedirs(temp_storage_path, exist_ok=True)

    # Read the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video file")

    # Extract frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame with proper naming convention (6 digits)
        frame_name = f"{frame_count+1:06d}.jpg"
        frame_path = os.path.join('img1', frame_name)
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

    # Backup original img1 folder
    if os.path.exists(original_img1_path):
        shutil.copytree(original_img1_path, os.path.join(temp_storage_path, 'img1_backup'))
        # Remove original img1
        shutil.rmtree(original_img1_path)
    
    # Move our newly created img1 to the original location
    shutil.move('img1', original_img1_path)

def restore_original_folder(original_img1_path, temp_storage_path):
    """
    Restore the original img1 folder after processing
    
    Args:
        original_img1_path (str): Path to the current img1 folder
        temp_storage_path (str): Path where the original img1 folder was stored
    """
    # Remove the current img1 folder
    if os.path.exists(original_img1_path):
        shutil.rmtree(original_img1_path)
    
    # Restore the original img1 folder
    backup_path = os.path.join(temp_storage_path, 'img1_backup')
    if os.path.exists(backup_path):
        shutil.move(backup_path, original_img1_path)
        
    # Clean up temp storage if empty
    if not os.listdir(temp_storage_path):
        os.rmdir(temp_storage_path)

# Example usage
if __name__ == "__main__":
    video_path = "path/to/your/video.mkv"
    original_img1_path = "path/to/original/img1"
    temp_storage_path = "path/to/temp/storage"

    try:
        # Extract frames and swap folders
        extract_frames_and_manage_folders(video_path, original_img1_path, temp_storage_path)
        
        # Your pipeline code goes here
        # ...
        
        # After pipeline finishes, restore original folder
        restore_original_folder(original_img1_path, temp_storage_path)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # In case of error, attempt to restore original folder
        restore_original_folder(original_img1_path, temp_storage_path) 