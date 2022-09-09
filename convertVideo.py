import moviepy.editor as moviepy

clip = moviepy.VideoFileClip("D:\\Production\\Dataset\\Dataset_Problem_4\\Crowd\\Sample_file_vehicle3.tts")

clip.write_videofile("myvideo.mp4")