import sqlite3
import cv2
import pandas as pd
import numpy as np
import os

# Step 1: Rotate image, keeping the eyes horizontal and padding to avoid cropping
def rotate_and_pad(image, left_eye_pos, right_eye_pos, filename):
    delta_x = right_eye_pos[0] - left_eye_pos[0]
    delta_y = right_eye_pos[1] - left_eye_pos[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))
    
    eye_center = ((left_eye_pos[0] + right_eye_pos[0]) // 2, (left_eye_pos[1] + right_eye_pos[1]) // 2)
    M = cv2.getRotationMatrix2D(eye_center, angle, 1)
    h, w = image.shape[:2]
    
    # Controlled padding (limit to 20% of original size)
    pad_size = int(max(h, w) * 0.01)
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    rotated_image = cv2.warpAffine(padded_image, M, (padded_image.shape[1], padded_image.shape[0]), flags=cv2.INTER_LINEAR)

    # Adjust eye positions after padding
    left_eye_pos_padded = (left_eye_pos[0] + pad_size, left_eye_pos[1] + pad_size)
    right_eye_pos_padded = (right_eye_pos[0] + pad_size, right_eye_pos[1] + pad_size)
    
    eye_center_padded = ((left_eye_pos_padded[0] + right_eye_pos_padded[0]) // 2, (left_eye_pos_padded[1] + right_eye_pos_padded[1]) // 2)
    
    # Save and print detailed debug information
    print(f"\n{filename}: Rotating image by {angle:.2f} degrees.")
    print(f"Image dimensions before padding: {w}x{h}, after padding: {rotated_image.shape[1]}x{rotated_image.shape[0]}")
    print(f"Padding applied: {pad_size}px on all sides.")
    print(f"Eye center after padding: {eye_center_padded}")

    # Return two values: the rotated image and the adjusted (padded) eye positions
    return rotated_image, left_eye_pos_padded, right_eye_pos_padded


# Step 2: Center the eyes in the frame
# Center the eyes by translating the midpoint of the eyes to the center of the image
def center_eyes(image, left_eye_pos, right_eye_pos, filename):
    # Midpoint of the eyes
    eye_center = (
        (left_eye_pos[0] + right_eye_pos[0]) // 2,
        (left_eye_pos[1] + right_eye_pos[1]) // 2
    )
    
    # Get the image center
    image_center = (image.shape[1] // 2, image.shape[0] // 2)
    
    # Translation needed to center the eyes
    delta_x = image_center[0] - eye_center[0]
    delta_y = image_center[1] - eye_center[1]
    
    # Create the translation matrix
    M = np.float32([[1, 0, delta_x], [0, 1, delta_y]])
    centered_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Save and print detailed debug information
    print(f"Eye center before translation: {eye_center}, Image center: {image_center}")
    print(f"Translation needed to center eyes: Δx={delta_x}, Δy={delta_y}")
    print(f"New eye center should be at image center.")

    return centered_image



# Step 3: Scale to maintain consistent eye distance, using padded eye positions
def resize_to_eye_distance(image, left_eye_pos_padded, right_eye_pos_padded, target_eye_dist, filename):
    current_eye_dist = np.sqrt((right_eye_pos_padded[0] - left_eye_pos_padded[0]) ** 2 + 
                               (right_eye_pos_padded[1] - left_eye_pos_padded[1]) ** 2)
    scale_factor = target_eye_dist / current_eye_dist
    
    # Ensure scale factor is reasonable (e.g., between 0.5x and 2x)
    scale_factor = min(max(scale_factor, 0.5), 2)
    
    h, w = image.shape[:2]
    new_size = (int(w * scale_factor), int(h * scale_factor))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    
    # Debugging info
    print(f"\n{filename}: Resizing image. Current eye distance: {current_eye_dist:.2f}, target eye distance: {target_eye_dist:.2f}, scale factor: {scale_factor:.2f}")
    print(f"Image resized to {new_size[0]}x{new_size[1]}")
    
    return resized_image


# Step 4: Crop to 1080x1920 preserving aspect ratio
def crop_to_target_size(image, target_size=(1080, 1920), filename=None):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    aspect_ratio = w / h
    target_aspect_ratio = target_w / target_h

    if aspect_ratio > target_aspect_ratio:
        # Image is wider than the target; crop the width
        new_w = int(h * target_aspect_ratio)
        start_x = max(0, (w - new_w) // 2)
        cropped_image = image[:, start_x:start_x + new_w]
        print(f"{filename}: Cropping width from {w} to {new_w}")
    else:
        # Image is taller than the target; crop the height
        new_h = int(w / target_aspect_ratio)
        start_y = max(0, (h - new_h) // 2)
        cropped_image = image[start_y:start_y + new_h, :]
        print(f"{filename}: Cropping height from {h} to {new_h}")
    
    # Final resize to target size if needed
    cropped_image = cv2.resize(cropped_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Save and print detailed debug information
    print(f"{filename}: Final crop and resize to {target_w}x{target_h}")
    return cropped_image


# Processing pipeline for each image
def process_image(row, target_eye_dist, output_size, output_dir, index):
    try:
        image_path = f"./photos/{row['filename']}"
        image = cv2.imread(image_path)
        if image is None:
            return None, f"Image {row['filename']} could not be loaded."
        
        filename = row['filename']
        left_eye_pos = (row['left_pupil_x'], row['left_pupil_y'])
        right_eye_pos = (row['right_pupil_x'], row['right_pupil_y'])
        
        # Step 1: Rotate and pad
        rotated_image, left_eye_pos_padded, right_eye_pos_padded = rotate_and_pad(image, left_eye_pos, right_eye_pos, filename)
        cv2.imwrite(os.path.join(output_dir, f"debug_{index}_rotated.jpg"), rotated_image)

        # Step 2: Center the eyes (use the padded eye positions now)
        centered_image = center_eyes(rotated_image, left_eye_pos_padded, right_eye_pos_padded, filename)
        cv2.imwrite(os.path.join(output_dir, f"debug_{index}_centered.jpg"), centered_image)

        # Step 3: Resize to maintain consistent eye distance
        # Call resize_to_eye_distance with padded eye positions
        resized_image = resize_to_eye_distance(centered_image, left_eye_pos_padded, right_eye_pos_padded, target_eye_dist, filename)
        cv2.imwrite(os.path.join(output_dir, f"debug_{index}_resized.jpg"), resized_image)
        
        return resized_image, None
    
        # Step 4: Crop to target size (1080x1920)
        # final_image = crop_to_target_size(resized_image, target_size=output_size, filename=filename)
        # cv2.imwrite(os.path.join(output_dir, f"debug_{index}_cropped.jpg"), final_image)
        
        # return final_image, None
    except Exception as e:
        return None, f"Error processing image {row['filename']}: {e}"

def main():
    output_size = (1080, 1920)  # Output portrait resolution (1080x1920)
    output_dir = './intermediate_images'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Connect to the SQLite database and retrieve eye positions
    conn = sqlite3.connect('app_data.db')
    df = pd.read_sql_query("SELECT * FROM eye_positions ORDER BY datetime_taken", conn)
    conn.close()

    # For debugging, only process the first 5 images
    df = df.head(5)

    # Calculate the maximum eye distance across all images
    df['eye_dist'] = np.sqrt((df['right_pupil_x'] - df['left_pupil_x']) ** 2 + (df['right_pupil_y'] - df['left_pupil_y']) ** 2)
    target_eye_dist = df['eye_dist'].max()

    # Process images sequentially for debugging
    for index, row in df.iterrows():
        print(f"\nProcessing image {index + 1}/{len(df)}: {row['filename']}")
        image, error = process_image(row, target_eye_dist, output_size, output_dir, index)
        if error:
            print(error)

if __name__ == "__main__":
    main()
