import cv2
import json
import numpy as np
import os
import subprocess

def assemble_summary(video_path, importance_scores, kts_segments, selected_indices, frame_interval, orig_frame_count, out_path="summary_video.mp4"):
    """
    Reads the original video, selectively writes frames belonging to 
    selected KTS segments based on knapsack optimization, and saves scores.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video.")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    temp_out_path = out_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))
    
    # Flatten selected segments into a set of frame indices
    selected_frames = set()
    for tr in selected_indices:
        # Avoid out of bounds
        if tr < len(kts_segments):
            start, end = kts_segments[tr]
            start_orig = start * frame_interval
            end_orig = min(end * frame_interval, orig_frame_count)
            for i in range(start_orig, end_orig):
                selected_frames.add(i)
                
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx in selected_frames:
            out.write(frame)
            
        frame_idx += 1
        
    cap.release()
    out.release()
    
    # Convert the file to H.264 properly so it plays in the Streamlit web UI
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', temp_out_path, '-vcodec', 'libx264', '-crf', '28', '-preset', 'fast', out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        # Rename fallback
        if os.path.exists(temp_out_path):
            os.rename(temp_out_path, out_path)
    
    return out_path
