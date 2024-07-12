from ultralytics import YOLO
import streamlit as st
import moviepy.editor as moviepy
import os
import tempfile


bestModelpath = '/mount/src/footballplayervid_detect/best.pt'
model = YOLO(bestModelpath)
switch = 0 # Switch caseleri kullanarak diğer ekranlar arası geçiş sağla


if 'button1' not in st.session_state:
    st.session_state.button1 = False

def click_button1():
    st.session_state.button1 = True

if not st.session_state.button1:
    st.title("Footballer Detection from Video!")
    st.button('Start the app.', on_click=click_button1)
else:
    uploaded_file = st.file_uploader(r"$\textsf{\Large Upload your video that is a section of any football match!}$", type=["mp4"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.title("Original Video")
        st.video(uploaded_file)
        if 'button2' not in st.session_state:
            st.session_state.button2 = False           
        def click_button2():
            st.session_state.button2 = True                                    
        if not st.session_state.button2:
            st.button('Show the processed video', on_click=click_button2)
        else:
            video = moviepy.VideoFileClip(video_path)
            total_frames = int(video.fps * video.duration)
            progress_bar = st.progress(0, text= "Processing the video. Please wait...")
            results = model.predict(source=video_path, save=True, project="results", name="predict", stream=True)
                    
            frame_id = 0
            directory = None
            for r in results:
                if frame_id == 0:
                    directory = r
                frame_id += 1
                progress_bar.progress(int((frame_id / total_frames) * 100), text="%" + str(int(((frame_id / total_frames) * 100))))
                    
            progress_bar.progress(100, text="Done!")
            video_name = os.path.splitext(os.path.basename(video_path))[0] + '.avi'
                    
            if directory is None:
                raise ValueError("Directory does not exist!")
                    
            results_path = directory.save_dir
                    
            saved_file_path = os.path.join(results_path, video_name)
            final_video_name = 'finalvideo.mp4'
            conv_file_path = os.path.join(results_path, final_video_name)
                    
            video2conv = moviepy.VideoFileClip(str(saved_file_path))
            video2conv.write_videofile(conv_file_path)

            st.title("Final Video")
                            
            final_video_path = conv_file_path
            final_video_file = open(final_video_path, 'rb')
            final_video_bytes = final_video_file.read()
                            
            st.video(final_video_bytes)                    
            st.success('Processing completed!')
 
        
       
                
                   
                            
            
