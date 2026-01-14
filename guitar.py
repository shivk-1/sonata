import cv2
import mediapipe as mp
import pygame
import numpy as np
import random
import math

#constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
BACKGROUND_COLOR = (15, 20, 40)  # A deep night blue

WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)

STRING_COLOR = (220, 220, 230)
STRING_ACTIVE_COLOR = (255, 167, 87)
KEY_PRESS_COLORS = [  # reused palette for particles
    (255, 87, 87), (255, 167, 87), (255, 250, 87),
    (87, 255, 127), (87, 187, 255)
]

DEBOUNCE_FRAMES = 3  # Number of frames a gesture must be held to register

PARTICLES = []  # global list

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-3, -0.5)
        self.lifespan = random.randint(20, 40)
        self.radius = random.randint(3, 7)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.05  # gravity
        self.lifespan -= 1
        self.radius -= 0.08
        if self.radius < 0: self.radius = 0

    def draw(self, screen):
        alpha = int(255 * (self.lifespan / 40))
        if alpha > 0:
            surface = pygame.Surface((max(1, int(self.radius * 2)), max(1, int(self.radius * 2))), pygame.SRCALPHA)
            pygame.draw.circle(surface, self.color + (alpha,), (int(self.radius), int(self.radius)), int(self.radius))
            screen.blit(surface, (self.x - self.radius, self.y - self.radius))

#karplus string synth 
def generate_guitar_wave(frequency, duration=1.2, sample_rate=44100, amplitude=0.6):
    """
    Simple Karplus-Strong pluck synthesis.
    Returns a pygame Sound object (stereo).
    """
    if frequency <= 0:
        #silence
        n_samples = int(duration * sample_rate)
        silence = np.zeros((n_samples, 2), dtype=np.int16)
        return pygame.sndarray.make_sound(silence)

    N = max(2, int(sample_rate // frequency))
    n_samples = int(duration * sample_rate)
    #buffer init
    buf = np.random.uniform(-1.0, 1.0, N)
    out = np.zeros(n_samples)

    #looped averaging filter
    idx = 0
    for i in range(n_samples):
        out[i] = buf[idx]
        #avg + decay
        next_idx = (idx + 1) % N
        avg = 0.5 * (buf[idx] + buf[next_idx])
        buf[idx] = 0.995 * avg  #decay factor
        idx = next_idx

    #pluck envelope
    envelope = np.concatenate((
        np.linspace(0, 1.0, int(0.01 * sample_rate)),  #attack
        np.exp(-3.0 * np.linspace(0, 1, n_samples - int(0.01 * sample_rate)))
    ))
    envelope = envelope[:n_samples]
    out *= envelope

    #normalize
    out = out / (np.max(np.abs(out)) + 1e-9)
    sound_data = (out * (2**15 - 1) * amplitude).astype(np.int16)
    stereo = np.ascontiguousarray(np.vstack((sound_data, sound_data)).T)
    return pygame.sndarray.make_sound(stereo)

#finger counting
def count_fingers(hand_landmarks, hand_label):
    if not hand_landmarks:
        return 0
    tip_ids = [4, 8, 12, 16, 20]
    fingers_up = 0
    #thumb
    if hand_label == 'Right':
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 2].x:
            fingers_up += 1
    else:
        if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 2].x:
            fingers_up += 1
    #other fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers_up += 1
    return fingers_up

#background
STARS = [(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), random.randint(1, 2)) for _ in range(150)]
def draw_background(screen):
    screen.fill(BACKGROUND_COLOR)
    for x, y, r in STARS:
        brightness = random.randint(100, 255)
        pygame.draw.circle(screen, (brightness, brightness, brightness), (x, y), r)

def draw_circular_webcam(screen, frame):
    frame_height, frame_width, _ = frame.shape
    cam_diameter = 300
    frame_small = cv2.resize(frame, (cam_diameter, cam_diameter))
    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    cam_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
    mask_surface = pygame.Surface((cam_diameter, cam_diameter), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (cam_diameter // 2, cam_diameter // 2), cam_diameter // 2)
    cam_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    pos_x = SCREEN_WIDTH // 2 - cam_diameter // 2
    pos_y = 150
    pygame.draw.circle(screen, KEY_PRESS_COLORS[4], (pos_x + cam_diameter//2, pos_y + cam_diameter//2), cam_diameter//2 + 8)
    screen.blit(cam_surface, (pos_x, pos_y))

#guitar drawing
def draw_guitar(screen, string_names, active_string_idx):
    
    neck_top = SCREEN_HEIGHT - 360
    neck_bottom = SCREEN_HEIGHT - 240
    neck_rect = pygame.Rect(80, neck_top, SCREEN_WIDTH - 160, neck_bottom - neck_top)
    pygame.draw.rect(screen, (40, 30, 25), neck_rect, border_radius=14)

    num_strings = len(string_names)
    spacing = (neck_rect.width - 40) / (num_strings - 1)
    start_x = neck_rect.x + 20
    for i, name in enumerate(string_names):
        x = int(start_x + i * spacing)
        
        thickness = 3 if i >= 4 else 2  # thicker for low strings
        color = STRING_ACTIVE_COLOR if (i == active_string_idx) else STRING_COLOR
        
        pygame.draw.line(screen, color, (x, neck_rect.y + 8), (x, neck_rect.bottom - 8), thickness)
        
        font = pygame.font.Font(None, 28)
        text = font.render(name, True, WHITE_COLOR)
        screen.blit(text, (x - text.get_width()//2, neck_rect.y - 28))

#main app
def run_guitar_app():
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Magical Hand Guitar")
    title_font = pygame.font.Font(None, 72)
    info_font = pygame.font.Font(None, 36)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    #tuning
    string_freqs = [82.4069, 110.0000, 146.8324, 196.0000, 246.9417, 329.6276]
    string_names = ["E2", "A2", "D3", "G3", "B3", "E4"]
    guitar_sounds = [generate_guitar_wave(f, duration=1.5) for f in string_freqs]

    #state
    stable_fingers = {'Left': 0, 'Right': 0}
    pending_fingers = {'Left': 0, 'Right': 0}
    debounce_counters = {'Left': 0, 'Right': 0}
    #last palm track
    last_palm_x = {'Left': 0.5, 'Right': 0.5}

    global PARTICLES

    running = True
    clock = pygame.time.Clock()

    while running:
        #events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        raw_fingers = {'Left': 0, 'Right': 0}
        active_string_idx = None

        #landmarks
        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handedness.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                raw_fingers[hand_label] = count_fingers(hand_landmarks, hand_label)
                #palm center
                palm_x = hand_landmarks.landmark[0].x
                last_palm_x[hand_label] = palm_x

        #debounce/mapping
        for hand_label in ['Left', 'Right']:
            if raw_fingers[hand_label] == pending_fingers[hand_label]:
                debounce_counters[hand_label] += 1
            else:
                pending_fingers[hand_label] = raw_fingers[hand_label]
                debounce_counters[hand_label] = 0

            if debounce_counters[hand_label] >= DEBOUNCE_FRAMES:
                if stable_fingers[hand_label] != pending_fingers[hand_label]:
                    stable_fingers[hand_label] = pending_fingers[hand_label]
                    #gesture response
                    if stable_fingers[hand_label] > 0:
                        #string based on palm
                        palm_x = last_palm_x.get(hand_label, 0.5)
                        idx = int(np.clip(palm_x * len(string_freqs), 0, len(string_freqs)-1))
                        #pluck
                        guitar_sounds[idx].play()
                        active_string_idx = idx
                        key_x_pos = 100 + idx * ((SCREEN_WIDTH - 200) / 6) + 70
                        for _ in range(24):
                            PARTICLES.append(Particle(key_x_pos + random.uniform(-10, 10),
                                                      SCREEN_HEIGHT - 300 + random.uniform(-20, 20),
                                                      KEY_PRESS_COLORS[idx % len(KEY_PRESS_COLORS)]))

        draw_background(screen)
        draw_guitar(screen, string_names, active_string_idx if active_string_idx is not None else -1)

        #particles
        for p in PARTICLES:
            p.update()
        PARTICLES = [p for p in PARTICLES if p.lifespan > 0]
        for p in PARTICLES:
            p.draw(screen)

        draw_circular_webcam(screen, frame)

        title_text = title_font.render("Magical Hand Guitar", True, WHITE_COLOR)
        info_l = info_font.render(f"Left fingers: {stable_fingers['Left']}", True, WHITE_COLOR)
        info_r = info_font.render(f"Right fingers: {stable_fingers['Right']}", True, WHITE_COLOR)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() / 2, 40))
        screen.blit(info_l, (100, 60))
        screen.blit(info_r, (SCREEN_WIDTH - 100 - info_r.get_width(), 60))

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()
    print("Application closed successfully.")

if __name__ == '__main__':
    run_guitar_app()
