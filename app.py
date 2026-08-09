import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

from utils.kts import kts_segmentation
from utils.knapsack import knapsack_ortools
from utils.assembly import assemble_summary
from data.video_utils import extract_frames
from data.extract_features import FeatureExtractor
from models.hisat import HiSAT

# Set up page config
st.set_page_config(page_title="HiSAT Video Summarization", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    border: none;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #ff3333;
    transform: scale(1.05);
}
.metric-card {
    background-color: #1e2127;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("HiSAT Video Summarizer")
st.markdown("Hierarchical Saliency-Attentive Transformer for robust video summarization")

st.sidebar.header("Configuration")
budget_pct = st.sidebar.slider("Summary Budget (%)", min_value=5, max_value=50, value=15, step=5)
model_choice = st.sidebar.selectbox("Model Structure", ["HiSAT (Recommended)", "Base Transformer"])

uploaded_file = st.sidebar.file_uploader(" UPLOAD VIDEO", type=["mp4", "avi", "mov"])

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = FeatureExtractor(device=device)
    # Initialize un-trained dummy weights for demonstration, in a real scenario we'd load weights here
    hisat = HiSAT().to(device)
    hisat.eval()
    return extractor, hisat, device

if uploaded_file is not None:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    # Calculate original video properties
    orig_cap = cv2.VideoCapture(video_path)
    orig_fps = orig_cap.get(cv2.CAP_PROP_FPS)
    orig_frame_count = int(orig_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_duration = orig_frame_count / orig_fps if orig_fps > 0 else 0
    orig_cap.release()
    orig_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    orig_duration_str = f"{int(orig_duration // 60):02d}:{int(orig_duration % 60):02d}"
    orig_size_str = f"{orig_size_mb:.2f} MB"
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.video(video_path)
    with col2:
        st.info("Video loaded successfully. Ready to generate summary.")
        st.metric("Duration", orig_duration_str)
        st.metric("Size", orig_size_str)
        generate_btn = st.button("Generate Summary", use_container_width=True)
    
    if 'summary_generated' not in st.session_state:
        st.session_state.summary_generated = False
        
    if generate_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading models...")
        extractor, model, device = load_models()
        progress_bar.progress(20)
        
        status_text.text("Extracting frames and finding shot boundaries...")
        frames_tensor, original_frames, frame_interval = extract_frames(video_path)
        N = frames_tensor.size(0)
        
        # Use basic uniform boundaries
        shot_boundaries = list(range(0, N, max(1, N // 10)))
        progress_bar.progress(40)
        
        status_text.text("Extracting semantic and saliency features...")
        F_sem = extractor.extract_semantic(frames_tensor)
        F_sal_spatial, s_scores = extractor.extract_saliency(frames_tensor)
        
        # Add batch dimension
        F_sem = F_sem.unsqueeze(0).to(device)
        F_sal_spatial = F_sal_spatial.unsqueeze(0).to(device)
        s_scores = s_scores.unsqueeze(0).to(device)
        progress_bar.progress(60)
        
        status_text.text("Running HiSAT Inference...")
        with torch.no_grad():
            pred_scores, pred_budget, h_temporal = model(F_sem, F_sal_spatial, s_scores, shot_boundaries)
            
        scores_np = pred_scores[0].cpu().numpy().flatten()
        saliency_np = s_scores[0].cpu().numpy().flatten()
        
        # Apply KTS and Knapsack
        kts_segments = kts_segmentation(h_temporal[0].cpu().numpy())
        
        # Aggregate score per segment
        seg_scores = []
        seg_lengths = []
        for start, end in kts_segments:
            if end > start:
                seg_scores.append(float(np.mean(scores_np[start:end])))
                seg_lengths.append(end - start)
            else:
                seg_scores.append(0.0)
                seg_lengths.append(0)
                
        budget_frames = int(N * (budget_pct / 100.0))
        selected_indices = knapsack_ortools(seg_lengths, seg_scores, budget_frames)
        
        # Create binary summary array
        machine_summary = np.zeros(N, dtype=int)
        for idx in selected_indices:
            start, end = kts_segments[idx]
            machine_summary[start:end] = 1
            
        progress_bar.progress(80)
        
        status_text.text("Assembling final video...")
        out_video_path = assemble_summary(video_path, scores_np, kts_segments, selected_indices, frame_interval, orig_frame_count)
        
        progress_bar.progress(100)
        status_text.empty()
        
        st.session_state.summary_generated = True
        st.session_state.out_video_path = out_video_path
        st.session_state.N = N
        st.session_state.machine_summary = machine_summary
        st.session_state.scores_np = scores_np
        st.session_state.saliency_np = saliency_np
        
        orig_cap = cv2.VideoCapture(video_path)
        orig_fps = orig_cap.get(cv2.CAP_PROP_FPS)
        orig_frame_count = int(orig_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_duration = orig_frame_count / orig_fps if orig_fps > 0 else 0
        orig_cap.release()
        orig_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        sum_cap = cv2.VideoCapture(out_video_path)
        sum_fps = sum_cap.get(cv2.CAP_PROP_FPS)
        sum_frame_count = int(sum_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sum_duration = sum_frame_count / sum_fps if sum_fps > 0 else 0
        sum_cap.release()
        sum_size_mb = os.path.getsize(out_video_path) / (1024 * 1024)
        
        st.session_state.orig_duration_str = f"{int(orig_duration // 60):02d}:{int(orig_duration % 60):02d}"
        st.session_state.sum_duration_str = f"{int(sum_duration // 60):02d}:{int(sum_duration % 60):02d}"
        st.session_state.orig_size_str = f"{orig_size_mb:.2f} MB"
        st.session_state.sum_size_str = f"{sum_size_mb:.2f} MB"

    if st.session_state.get('summary_generated', False):
        st.success("Summary Generated Successfully!")
        
        # Top Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Original Duration", st.session_state.orig_duration_str)
        with m2:
            st.metric("Summary Duration", st.session_state.sum_duration_str)
        with m3:
            st.metric("Original Size", st.session_state.orig_size_str)
        with m4:
            st.metric("Summary Size", st.session_state.sum_size_str)
        with m5:
            st.metric("Data Saved", f"{float(st.session_state.orig_size_str.split()[0]) - float(st.session_state.sum_size_str.split()[0]):.2f} MB")
            
        st.markdown("---")
        
        st.subheader("Features Analysis")
        tab1, tab2 = st.tabs(["Predicted Importance", "Intrinsic Saliency Features"])
        
        plt.style.use('dark_background')
        
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(st.session_state.scores_np, color='#ff4b4b', linewidth=2)
            ax1.fill_between(np.arange(len(st.session_state.scores_np)), st.session_state.scores_np, alpha=0.3, color='#ff4b4b')
            ax1.set_title('Predicted Frame Importance Scores', fontsize=14, color='white')
            ax1.set_xlabel('Frame Index', fontsize=12, color='lightgray')
            ax1.set_ylabel('Score', fontsize=12, color='lightgray')
            ax1.grid(True, linestyle='--', alpha=0.3)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            fig1.patch.set_facecolor('#0e1117')
            ax1.set_facecolor('#0e1117')
            st.pyplot(fig1)
            
        with tab2:
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(st.session_state.saliency_np, color='#00ff00', linewidth=2)
            ax2.fill_between(np.arange(len(st.session_state.saliency_np)), st.session_state.saliency_np, alpha=0.2, color='#00ff00')
            ax2.set_title('Intrinsic Saliency Features', fontsize=14, color='white')
            ax2.set_xlabel('Frame Index', fontsize=12, color='lightgray')
            ax2.set_ylabel('Saliency', fontsize=12, color='lightgray')
            ax2.grid(True, linestyle='--', alpha=0.3)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            fig2.patch.set_facecolor('#0e1117')
            ax2.set_facecolor('#0e1117')
            st.pyplot(fig2)
        
        st.markdown("---")
        st.subheader("🎥 Final Summarized Video")
        
        res_col1, res_col2 = st.columns([2, 1])
        with res_col1:
            st.video(st.session_state.out_video_path)
            
        with res_col2:
            st.metric("Duration", st.session_state.sum_duration_str)
            st.metric("Size", st.session_state.sum_size_str)
            st.markdown("### Export")
            with open(st.session_state.out_video_path, "rb") as fp:
                st.download_button("Download Video", fp, file_name="summary_video.mp4", mime="video/mp4", use_container_width=True)

    # Cleanup the original video temporary file if needed, but only if we're done
    # Actually streamlit re-runs the whole script, so keeping the tfile could cause leaks if not careful.
    # We will just let the OS handle temp files for this simple app.
