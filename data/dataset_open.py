import h5py

# Replace with 'SumMe.h5' or your specific filename
filename = str(input("Pls"))

with h5py.File(filename, 'r') as f:
    # 1. List all video IDs in the file
    video_ids = list(f.keys())
    print(f"Total videos: {len(video_ids)}")
    
    # 2. Access the first video's data
    if video_ids:
        first_video = video_ids[0]
        print(f"\nData for video: {first_video}")
        
        # List datasets inside this video (features, gtscore, etc.)
        for key in f[first_video].keys():
            data = f[first_video][key][()]  # Use [()] to load the data into a NumPy array
            print(f" - {key}: shape {data.shape}")
