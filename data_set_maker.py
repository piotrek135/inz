import cv2
import numpy as np
import pygame
import sys
import time

def save_label_and_image(rects):
    to_write = ''

    for r in rects:
        pos1 = r[0]
        pos2 = r[1]
        specie = r[2]

        to_write = to_write + str(str(specie) + ' ' 
            + str((pos1[0]+pos2[0])/(2*window_size[0])) + ' '
            + str((pos1[1]+pos2[1])/(2*window_size[1])) + ' '
            + str(abs(pos1[0]-pos2[0])/window_size[0]) + ' '
            + str(abs(pos1[1]-pos2[1])/window_size[1]) + '\n')
    
    file_name = name_mapping.get(rects[0][2]) + '_' + f'{count[rects[0][2]]}'
    full_save_path = save_path + name_mapping.get(rects[0][2]) + '\\'

    count[rects[0][2]] = count[rects[0][2]] + 1

    with open(full_save_path + file_name + '.txt', 'w') as file:
        file.write(to_write)

    cv2.imwrite(full_save_path + file_name + '.jpg', frame_og)
    return

color_mapping = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255)
}

name_mapping = {
    0: "Sikorka",
    1: "Rudzik",
    2: "Sojka"
}

save_path = "F:\\zzzz STUDIA\\sem7\\inz\\test\\"
count = [684,5,16]

pygame.init()

window_size = (1280, 720)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Video Annotation")

white = (255, 255, 255)
red = (255, 0, 0)

video_path = "F:\\zzzz STUDIA\\sem7\\inz\\all_zla_etykieta.mp4"
cap = cv2.VideoCapture(video_path)

playing = False
current_frame = 0
rects = []
texts_to_draw = []
rect_start = None
rect_end = None
mouse_x = 0
mouse_y = 0
specie = 0
last_frame_time = time.time()
time_delay = 0.1

while True:
    ret, frame_og = cap.read()

    if not ret:
        break
    frame = frame_og

    frame = cv2.resize(frame, window_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    pygame_frame = pygame.surfarray.make_surface(frame)

    if rect_start is not None and rect_end is not None:
        pygame.draw.rect(pygame_frame, color_mapping.get(specie), (rect_start, rect_end - rect_start), 2)
    
    if rects != []:
        for r in rects:
            pygame.draw.rect(pygame_frame, color_mapping.get(r[2]), (r[0], r[1] - r[0]), 2)
            text = pygame.font.SysFont('calibri', 24).render(f"{name_mapping.get(r[2])}", True, (255,255,255))
            text_rect = text.get_rect()
            text_rect.center = (r[0][0], r[0][1] - 10)
            text_rect.left = r[0][0]
            pygame.draw.rect(pygame_frame, color_mapping.get(r[2]), text_rect)
            texts_to_draw.append([text, text_rect])

    pygame.draw.line(pygame_frame, red, (mouse_x, 0), (mouse_x, window_size[1]), 2)
    pygame.draw.line(pygame_frame, red, (0, mouse_y), (window_size[0], mouse_y), 2)

    screen.blit(pygame_frame, (0, 0))

    text = pygame.font.SysFont('calibri', 24).render(f"{name_mapping.get(specie)}", True, (255,255,255))
    text_rect = text.get_rect()
    text_rect.center = (mouse_x, mouse_y - 10)
    text_rect.left = mouse_x
    screen.blit(text,text_rect)

    text = pygame.font.SysFont('calibri', 32).render(f"{current_frame}", True, (255,255,255))
    text_rect = text.get_rect()
    text_rect.center = (30, 30)
    screen.blit(text,text_rect)

    if texts_to_draw != []:
        for t in texts_to_draw:
            screen.blit(t[0], t[1])
    texts_to_draw = []

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                playing = not playing

            elif event.key == pygame.K_RIGHT:
                current_frame += 1

            elif event.key == pygame.K_LEFT:
                current_frame = max(0, current_frame - 1)
            
            elif event.key == pygame.K_KP_PLUS:
                if time_delay > 0:
                    time_delay -= 0.1
            
            elif event.key == pygame.K_KP_MINUS:
                
                time_delay += 0.1

            elif pygame.K_1 <= event.key <= pygame.K_5:
                specie = event.key - pygame.K_1
            
            elif event.key == pygame.K_BACKSPACE:
                if rects != []:
                    rects.pop()
                if texts_to_draw != []:
                    texts_to_draw.pop(len(texts_to_draw)- 1)
                    print(texts_to_draw)

            elif event.key == pygame.K_RETURN:
                if rects != []:
                    save_label_and_image(rects)
                    rects = []
                    texts_to_draw = []
                current_frame += 1

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                rect_start = np.array(event.pos)
                rect_end = None

        elif event.type == pygame.MOUSEMOTION:
            if rect_start is not None:
                rect_end = np.array(event.pos)

            mouse_x, mouse_y = event.pos

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and rect_start is not None:
                rect_end = np.array(event.pos)

                rects.append([rect_start, rect_end, specie])
                rect_start = None
                rect_end = None
                

    current_time = time.time()
    if playing and current_time - last_frame_time > time_delay:
        last_frame_time = current_time
        current_frame += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

cap.release()
pygame.quit()