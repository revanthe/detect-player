import streamlit as st
import tempfile
import os
import moviepy.editor as moviepy
from ultralytics import YOLO

st.title('Detection of Players at Sports Events')
best_model_path = r"C:\Users\cenke\Image-process\Ready2Deploy\best.pt"
model = YOLO(best_model_path)

def set_state():
    if 'stage' not in st.session_state:
        st.session_state.stage = 0
    else:
        st.session_state.stage += 1
    st.experimental_rerun()

# İlk başta stage'i ayarlayın
if 'stage' not in st.session_state:
    st.session_state.stage = 0

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'video_path' not in st.session_state:
    st.session_state.video_path = None

if 'directory' not in st.session_state:
    st.session_state.directory = None

if 'conv_file_path' not in st.session_state:
    st.session_state.conv_file_path = None

match st.session_state.stage:
    case 0:
        if st.button('Upload your video...'):
            set_state()
    case 1:
        st.session_state.uploaded_file = st.file_uploader(r"$\textsf{\Large Upload your video that is a section of any football match!}$", type=["mp4"])
        st.button('Show original video...', on_click=set_state())
    case 2:
        if st.session_state.uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tfile.write(st.session_state.uploaded_file.read())
            st.session_state.video_path = tfile.name
            st.title("Original Video")
            st.video(st.session_state.uploaded_file)
            st.button('Process the Video...',on_click=set_state())
    case 3:
        if st.session_state.video_path is not None and st.session_state.conv_file_path is None:
            video = moviepy.VideoFileClip(st.session_state.video_path)
            total_frames = int(video.fps * video.duration)
            progress_bar = st.progress(0, text="Processing the video. Please wait...")
            results = model.predict(source=st.session_state.video_path, save=True, project="results", name="predict", stream=True)
            frame_id = 0
            for r in results:
                if frame_id == 0:
                    st.session_state.directory = r
                frame_id += 1
                progress_bar.progress(int((frame_id / total_frames) * 100), text="%" + str(int(((frame_id / total_frames) * 100))))
            progress_bar.progress(100, text="Done!")
            video_name = os.path.splitext(os.path.basename(st.session_state.video_path))[0] + '.avi'
            if st.session_state.directory is None:
                raise ValueError("Directory does not exist!")
            results_path = st.session_state.directory.save_dir
            saved_file_path = os.path.join(results_path, video_name)
            final_video_name = 'finalvideo.mp4'
            st.session_state.conv_file_path = os.path.join(results_path, final_video_name)
            video2conv = moviepy.VideoFileClip(str(saved_file_path))
            video2conv.write_videofile(st.session_state.conv_file_path)
            st.success('Processing completed!')
            st.button('Show the Processed video...', on_click=set_state())
    case 4:
        if st.session_state.conv_file_path is not None:
            st.title("Final Video")
            final_video_file = open(st.session_state.conv_file_path, 'rb')
            final_video_bytes = final_video_file.read()
            st.video(final_video_bytes)

