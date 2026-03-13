#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
from ultralytics import YOLO
from PIL import Image
import cv2
import shutil
import requests

# Import Roboflow SDK
from roboflow import Roboflow


def get_user_input(prompt, default=None):
    """Get user input with default value support"""
    if default:
        user_input = input(f"{prompt} (default: {default}): ").strip()
        return user_input if user_input else default
    else:
        user_input = input(f"{prompt}: ").strip()
        return user_input


def simple_roboflow_download(api_key):
    print("\n=== Roboflow Download ===")

    # Get basic information
    workspace_id = "mtsu-2h73y"  # Use known workspace directly
    project_id = get_user_input("Project ID (e.g.: weed1-rmxbe)")
    version_num = get_user_input("Version number", "1")
    download_format = get_user_input("Download format (yolov8/yolov11/coco/voc)", "yolov8")
    download_path = get_user_input("Download path", f"./downloads/{project_id}")

    print(f"\nStarting download...")
    print(f"  Workspace: {workspace_id}")
    print(f"  Project: {project_id}")
    print(f"  Version: {version_num}")
    print(f"  Format: {download_format}")
    print(f"  Path: {download_path}")

    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace_id).project(project_id)
        dataset = project.version(int(version_num)).download(download_format, location=download_path)

        print(f"Download successful: {download_path}")
        return download_path

    except Exception as e:
        print(f"Download failed: {e}")

        # Provide common solution suggestions
        print("\nPossible solutions:")
        print("1. Try different version numbers (1, 2, 3...)")
        print("2. Check if project ID is correct")
        print("3. Try different download formats")
        print("4. Check network connection")

        return None


def auto_label_images():
    """Auto-label images"""
    print("\n=== Auto-label Images ===")

    # User input configuration
    model_path = get_user_input("YOLO model path", "bestweed.pt")
    source_folder = get_user_input("Source image folder", "nonlable")
    output_folder = get_user_input("Output folder name", "auto_labeled")
    confidence_threshold = float(get_user_input("Confidence threshold", "0.25"))

    print(f"\nConfiguration:")
    print(f"  Model: {model_path}")
    print(f"  Source folder: {source_folder}")
    print(f"  Output folder: {output_folder}")
    print(f"  Confidence threshold: {confidence_threshold}")

    # Check files and folders
    if not os.path.exists(model_path):
        print(f"Model file does not exist: {model_path}")
        return None, None, None

    if not os.path.exists(source_folder):
        print(f"Source folder does not exist: {source_folder}")
        return None, None, None

    # Create output directory structure
    detected_images = os.path.join(output_folder, "detected", "images")
    detected_labels = os.path.join(output_folder, "detected", "labels")
    no_detection_images = os.path.join(output_folder, "no_detection", "images")
    no_detection_labels = os.path.join(output_folder, "no_detection", "labels")

    for folder in [detected_images, detected_labels, no_detection_images, no_detection_labels]:
        os.makedirs(folder, exist_ok=True)

    print("Output directory structure created")

    # Load YOLO model
    print(f"\nLoading YOLO model: {model_path}")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None, None, None

    # Collect image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_folder, ext)))
    image_files.sort()
    total_images = len(image_files)

    print(f"\nFound {total_images} images")
    if total_images == 0:
        print("No image files found")
        return None, None, None

    # Start auto-labeling
    processed_count = 0
    detected_count = 0
    no_detection_count = 0

    print("\nStarting auto-labeling process...")

    for i, image_path in enumerate(image_files):
        try:
            image_name = os.path.basename(image_path)
            image_name_no_ext = os.path.splitext(image_name)[0]

            # YOLO inference
            results = model.predict(
                source=image_path,
                conf=confidence_threshold,
                save=False,
                verbose=False
            )

            img = cv2.imread(image_path)
            if img is None:
                print(f"  Cannot read image: {image_name}")
                continue

            img_height, img_width = img.shape[:2]

            detections = []
            has_detection = False

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    has_detection = True
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        x_center = (x1 + x2) / 2 / img_width
                        y_center = (y1 + y2) / 2 / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        detection_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        detections.append(detection_line)

            # Save results
            if has_detection:
                target_image_path = os.path.join(detected_images, image_name)
                target_label_path = os.path.join(detected_labels, f"{image_name_no_ext}.txt")
                shutil.copy2(image_path, target_image_path)
                with open(target_label_path, 'w') as f:
                    for detection in detections:
                        f.write(detection + '\n')
                detected_count += 1
            else:
                target_image_path = os.path.join(no_detection_images, image_name)
                target_label_path = os.path.join(no_detection_labels, f"{image_name_no_ext}.txt")
                shutil.copy2(image_path, target_image_path)
                with open(target_label_path, 'w') as f:
                    f.write("")
                no_detection_count += 1

            processed_count += 1

            # Show progress
            if processed_count % 50 == 0 or processed_count == total_images:
                print(
                    f"  Progress: {processed_count}/{total_images} | Detected: {detected_count} | No detection: {no_detection_count}")

        except Exception as e:
            print(f"  Error processing image {image_name}: {e}")
            continue

    print(f"\nAuto-labeling completed!")
    print(f"  Total processed: {processed_count}")
    print(f"  Objects detected: {detected_count}")
    print(f"  No detection: {no_detection_count}")

    return output_folder, detected_count, os.path.join(output_folder, "detected")


def upload_to_roboflow(dataset_path, api_key, project_name):
    """Upload dataset to Roboflow"""
    print(f"\nUploading to Roboflow project: {project_name}")

    try:
        rf = Roboflow(api_key=api_key)

        # Get workspace
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.roboflow.com/", headers=headers)

        if response.status_code != 200:
            print("Unable to get workspace information")
            return False

        data = response.json()
        workspace_id = data.get('workspace')
        workspace = rf.workspace(workspace_id)

        print(f"Workspace: {workspace_id}")
        print(f"Target project: {project_name}")

        # Check dataset structure
        images_path = os.path.join(dataset_path, "images")
        labels_path = os.path.join(dataset_path, "labels")

        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"Invalid dataset structure")
            print(f"  Expected: {dataset_path}/images/ and {dataset_path}/labels/")
            return False

        # Count files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(glob.glob(os.path.join(images_path, ext)))

        label_files = glob.glob(os.path.join(labels_path, "*.txt"))

        print(f"Dataset information:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")

        # Upload dataset
        workspace.upload_dataset(
            dataset_path,
            project_name,
            num_workers=10,
            project_type="object-detection",
            batch_name="auto-labeled-batch",
            num_retries=3
        )

        print(f"Upload successful!")
        print(f"Project link: https://app.roboflow.com/{workspace_id}/{project_name}")
        return True

    except Exception as e:
        print(f"Upload failed: {e}")
        return False


def main_menu():
    """Main menu"""
    print("=" * 60)
    print("    YOLO Auto-Labeling and Roboflow Data Management Tool")
    print("=" * 60)
    print()
    print("Please select an option:")
    print("1. Download dataset from Roboflow")
    print("2. Auto-label local images and upload to Roboflow")
    print("3. Exit")
    print()

    choice = get_user_input("Select option (1-3)")
    return choice


def main():
    """Main program"""
    # Load API key from file (same as roboflow_bridge.py)
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".roboflow_key")
    if os.path.exists(key_file):
        with open(key_file) as f:
            API_KEY = f.read().strip()
    else:
        API_KEY = input("Enter your Roboflow API key: ").strip()
        with open(key_file, 'w') as f:
            f.write(API_KEY)
        os.chmod(key_file, 0o600)
        print(f"Key saved to {key_file}")

    while True:
        choice = main_menu()

        if choice == "1":
            # Download dataset
            download_path = simple_roboflow_download(API_KEY)
            if download_path:
                print(f"\nDataset downloaded to: {download_path}")
            input("\nPress Enter to continue...")

        elif choice == "2":
            # Auto-label and upload
            output_folder, detected_count, detected_path = auto_label_images()

            if output_folder and detected_count > 0:
                print(f"\nDetected {detected_count} images containing objects")

                upload_choice = get_user_input("Upload to Roboflow? (y/n)", "y")

                if upload_choice.lower() in ['y', 'yes']:
                    project_name = get_user_input("Enter target project name")

                    if project_name:
                        success = upload_to_roboflow(detected_path, API_KEY, project_name)
                        if success:
                            print("Complete workflow executed successfully!")
                        else:
                            print("Upload failed, but local files are saved")
                    else:
                        print("Project name cannot be empty")
                else:
                    print(f"Labeling results saved to: {output_folder}")
            elif output_folder:
                print("No images with detected objects found, no upload needed")

            input("\nPress Enter to continue...")

        elif choice == "3":
            print("Exiting program")
            break

        else:
            print("Invalid selection, please try again")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()