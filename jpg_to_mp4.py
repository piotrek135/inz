import cv2
import os

def create_video(image_folder, video_name='output.mp4', fps=10):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]

    #images.sort(key=lambda x: int(x.split('.')[0]))

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    for image in images:
        img_path = os.path.join(image_folder, image)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 360))
 
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

image_folder = 'F:\\zzzz STUDIA\\sem7\\inz\\pics\\'

video_name = 'video.mp4'

fps = 10

create_video(image_folder, video_name, fps)