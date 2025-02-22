import streamlit as st
import cv2
import numpy as np
import random
import os
import json
import time

# -------------------- Set Page Config --------------------
st.set_page_config(page_title="Guess the Image Game", layout="wide")

# -------------------- Helper Functions --------------------

def rotate_image(image, angle):
    """
    Rotate the image by the given angle while adjusting the image size
    so that the entire rotated image fits.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    rotated = cv2.warpAffine(image, M, (nW, nH))
    return rotated

def process_quadrant(quadrant, block_grid=3, blur_ksize=21):
    """
    Split a quadrant into (block_grid x block_grid) blocks,
    blur each block, randomly shuffle the blocks, and then
    reassemble them into a new quadrant.
    """
    (qh, qw) = quadrant.shape[:2]
    block_h = qh // block_grid
    block_w = qw // block_grid
    blocks = []
    
    for i in range(block_grid):
        for j in range(block_grid):
            y1 = i * block_h
            x1 = j * block_w
            block = quadrant[y1:(y1 + block_h), x1:(x1 + block_w)].copy()
            block = cv2.GaussianBlur(block, (blur_ksize, blur_ksize), 0)
            blocks.append(block)
    
    random.shuffle(blocks)
    
    new_quadrant = np.zeros_like(quadrant)
    idx = 0
    for i in range(block_grid):
        for j in range(block_grid):
            y1 = i * block_h
            x1 = j * block_w
            new_quadrant[y1:(y1 + block_h), x1:(x1 + block_w)] = blocks[idx]
            idx += 1
    return new_quadrant

def distort_image(image):
    """
    Distort image by rotating, splitting into quadrants,
    processing each quadrant (splitting into blocks, blurring, shuffling)
    and reassembling.
    """
    rotated = rotate_image(image, random.randint(-10, 10))
    (h, w) = rotated.shape[:2]
    mid_h = h // 2
    mid_w = w // 2
    top_left = rotated[0:mid_h, 0:mid_w]
    top_right = rotated[0:mid_h, mid_w:w]
    bottom_left = rotated[mid_h:h, 0:mid_w]
    bottom_right = rotated[mid_h:h, mid_w:w]
    
    block_grid = 2
    blur_ksize = 21
    processed_tl = process_quadrant(top_left, block_grid, blur_ksize)
    processed_tr = process_quadrant(top_right, block_grid, blur_ksize)
    processed_bl = process_quadrant(bottom_left, block_grid, blur_ksize)
    processed_br = process_quadrant(bottom_right, block_grid, blur_ksize)
    
    top_row = np.hstack((processed_tl, processed_tr))
    bottom_row = np.hstack((processed_bl, processed_br))
    final_image = np.vstack((top_row, bottom_row))
    return final_image

def format_elapsed_time(elapsed):
    """Format elapsed time (in seconds) to MM:SS:ms format."""
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    msecs = int((elapsed - int(elapsed)) * 1000)
    return f"{mins:02d}:{secs:02d}:{msecs:03d}"

# -------------------- Paths & Data Loading --------------------

# Folder with available images
IMAGE_FOLDER = r"\images_v7"
# Folder for used images
USED_FOLDER = r"\used_images"
os.makedirs(USED_FOLDER, exist_ok=True)

# JSON file with personality summaries
SUMMARY_JSON = r"H:\fun friday\personality_summaries_v1.json"
if os.path.exists(SUMMARY_JSON):
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summaries = json.load(f)
else:
    summaries = {}

# -------------------- Initialize Session State --------------------
if "current_image_file" not in st.session_state:
    st.session_state.current_image_file = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "distorted_image" not in st.session_state:
    st.session_state.distorted_image = None
if "original_shown" not in st.session_state:
    st.session_state.original_shown = False

# -------------------- Streamlit Layout --------------------
st.title("Guess the Image Game")

# Two columns for displaying images side by side
col_blur, col_orig = st.columns(2)

# --- Left Column: Blurred Image Section ---
with col_blur:
    st.header("Blurred Image")
    if st.button("Load Blurred Image", key="load_blur"):
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if image_files:
            selected_file = random.choice(image_files)
            st.session_state.current_image_file = selected_file
            st.session_state.start_time = time.time()
            st.session_state.original_shown = False  # Reset flag
            image_path = os.path.join(IMAGE_FOLDER, selected_file)
            image = cv2.imread(image_path)
            if image is not None:
                distorted = distort_image(image)
                st.session_state.distorted_image = cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB)
                st.image(st.session_state.distorted_image, caption="Blurred/Distorted Image", width=400)
            else:
                st.error("Error: Could not load the image.")
        else:
            st.error("No images found in the image folder.")

    # Ensure blurred image persists
    elif st.session_state.get("distorted_image") is not None:
        st.image(st.session_state.distorted_image, caption="Blurred/Distorted Image", width=400)


# --- Right Column: Original Image Section ---
with col_orig:
    st.header("Original Image")
    if st.button("Show Original Image", key="show_orig"):
        if st.session_state.current_image_file is None:
            st.warning("Please load a blurred image first.")
        else:
            elapsed = time.time() - st.session_state.start_time
            formatted_time = format_elapsed_time(elapsed)
            selected_file = st.session_state.current_image_file
            orig_image_path = os.path.join(IMAGE_FOLDER, selected_file)
            original = cv2.imread(orig_image_path)
            if original is not None:
                st.session_state.original_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                st.image(st.session_state.original_image, caption="Original Image", width=400)
                st.markdown(f"**Time taken to reveal:** {formatted_time}")
                personality_name = os.path.splitext(selected_file)[0].replace("_", " ")
                st.subheader(f"About {personality_name}")
                description = summaries.get(personality_name, "Description not found.")
                st.write(description)
                st.session_state.original_shown = True
            else:
                st.error("Error: Could not load the original image.")

    # Ensure original image persists
    elif st.session_state.get("original_image") is not None:
        st.image(st.session_state.original_image, caption="Original Image", width=400)

# --- Next Button Section ---
if st.session_state.original_shown:
    if st.button("Next"):
        # Move the current image to USED_FOLDER if it exists in IMAGE_FOLDER
        selected_file = st.session_state.current_image_file
        if selected_file:
            orig_image_path = os.path.join(IMAGE_FOLDER, selected_file)
            dest_path = os.path.join(USED_FOLDER, selected_file)
            try:
                if os.path.exists(orig_image_path):
                    os.rename(orig_image_path, dest_path)
                    st.success(f"Image {selected_file} moved to used images.")
                else:
                    st.info("Image was already moved.")
            except Exception as e:
                st.error(f"Error moving file: {e}")

        # Reset session state for next round (clearing images)
        st.session_state.current_image_file = None
        st.session_state.start_time = None
        st.session_state.distorted_image = None  # Clear blurred image
        st.session_state.original_image = None   # Clear original image
        st.session_state.original_shown = False
        st.rerun()  # Force refresh the page

# -------------------- Reset Game Section --------------------
st.write("---")
st.subheader("Reset Game")
with st.form("reset_form"):
    password = st.text_input("Enter password to reset game", type="password")
    submit_reset = st.form_submit_button("Reset Game")
    if submit_reset:
        if password == "1234":
            used_files = os.listdir(USED_FOLDER)
            for f in used_files:
                src = os.path.join(USED_FOLDER, f)
                dest = os.path.join(IMAGE_FOLDER, f)
                try:
                    os.rename(src, dest)
                except Exception as e:
                    st.error(f"Error resetting file {f}: {e}")
            st.success("Game has been reset. All used images moved back.")
        else:
            st.error("Incorrect password. Reset failed.")
