import cv2
import mediapipe as mp
import pygame
import numpy as np
import random
import math
import time

#constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
BACKGROUND_COLOR = (15, 20, 40)

WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
KEY_PRESS_COLORS = [
    (255, 87, 87), (255, 167, 87), (255, 250, 87),
    (87, 255, 127), (87, 187, 255)
]

#gameplay
DEBOUNCE_FRAMES = 3

#particles
PARTICLES = []

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-2.5, 2.5)
        self.vy = random.uniform(-5, -1)
        self.lifespan = random.randint(20, 42)
        self.radius = random.randint(3, 7)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.12
        self.lifespan -= 1
        self.radius -= 0.08
        if self.radius < 0:
            self.radius = 0

    def draw(self, screen):
        alpha = int(255 * (self.lifespan / 42))
        if alpha > 0 and self.radius > 0:
            surface = pygame.Surface((int(self.radius * 2), int(self.radius * 2)), pygame.SRCALPHA)
            pygame.draw.circle(surface, self.color + (alpha,), (int(self.radius), int(self.radius)), int(self.radius))
            screen.blit(surface, (self.x - self.radius, self.y - self.radius))

#sound gen
def make_stereo(sound_mono):
    if sound_mono.ndim == 1:
        return np.ascontiguousarray(np.vstack((sound_mono, sound_mono)).T)
    return sound_mono

#trombone
def generate_trombone_wave(base_freq, duration=1.0, sample_rate=44100, amplitude=0.6, slide_cent=0):
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    #pitch slide
    cents = np.linspace(0, slide_cent, n)
    freq_multiplier = 2 ** (cents / 1200.0)
    instantaneous_freq = base_freq * freq_multiplier

    #phase
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sample_rate

    #harmonics
    wave = 0.0
    harmonics = [1.0, 0.56, 0.28, 0.14, 0.08]  # trombone-ish harmonic balance
    for h_amp, h_mult in zip(harmonics, [1, 2, 3, 4, 5]):
        wave += h_amp * np.sin(h_mult * phase)

    #attack/decay
    attack_time = max(0.02, 0.08 * (1 - amplitude))
    env = np.ones_like(t)
    attack_samples = int(sample_rate * attack_time)
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1.0, attack_samples)
    env *= np.exp(-2.2 * t)

    #vibrato
    vibrato = 1 + 0.0025 * np.sin(2 * np.pi * 5.2 * t)
    wave *= env * vibrato

    #smoothing
    kernel = np.array([0.25, 0.5, 0.25])
    wave = np.convolve(wave, kernel, mode='same')

    #normalize
    if np.max(np.abs(wave)) > 0:
        wave = wave / np.max(np.abs(wave))
    wave = (wave * amplitude * (2**15 - 1)).astype(np.int16)

    stereo = make_stereo(wave)
    return pygame.sndarray.make_sound(stereo)

#finger counting
def count_fingers(hand_landmarks, hand_label):
    if not hand_landmarks:
        return 0
    tip_ids = [4, 8, 12, 16, 20]
    fingers_up = 0
    #thumb
    try:
        if hand_label == 'Right':
            if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 2].x:
                fingers_up += 1
        else:
            if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 2].x:
                fingers_up += 1
    except:
        pass
    #other fingers
    for i in range(1, 5):
        try:
            if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
                fingers_up += 1
        except:
            pass
    return fingers_up

#drawing
STARS = [(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), random.randint(1, 2)) for _ in range(150)]

def draw_background(screen):
    screen.fill(BACKGROUND_COLOR)
    for x, y, r in STARS:
        brightness = random.randint(110, 255)
        pygame.draw.circle(screen, (brightness, brightness, brightness), (x, y), r)

def draw_circular_webcam(screen, frame):
    cam_diameter = 300
    frame_small = cv2.resize(frame, (cam_diameter, cam_diameter))
    frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    cam_surface = pygame.surfarray.make_surface(np.rot90(frame_small))
    mask_surface = pygame.Surface((cam_diameter, cam_diameter), pygame.SRCALPHA)
    pygame.draw.circle(mask_surface, (255, 255, 255, 255), (cam_diameter // 2, cam_diameter // 2), cam_diameter // 2)
    cam_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    pos_x = SCREEN_WIDTH // 2 - cam_diameter // 2
    pos_y = 150
    pygame.draw.circle(screen, KEY_PRESS_COLORS[4], (pos_x + cam_diameter // 2, pos_y + cam_diameter // 2), cam_diameter // 2 + 8)
    screen.blit(cam_surface, (pos_x, pos_y))

def draw_trombone_ui(screen, note_names, active_note):
    #note strip
    strip_w = SCREEN_WIDTH - 200
    start_x = 100
    y = SCREEN_HEIGHT - 180
    key_w = strip_w / len(note_names)
    font = pygame.font.Font(None, 30)
    for i, n in enumerate(note_names):
        rect = pygame.Rect(int(start_x + i * key_w), y, int(key_w - 8), 110)
        color = KEY_PRESS_COLORS[i % len(KEY_PRESS_COLORS)] if n == active_note else (240, 240, 245)
        pygame.draw.rect(screen, (35, 35, 35), rect.move(0, 6), border_radius=12)  # shadow
        pygame.draw.rect(screen, color, rect, border_radius=12)
        label = font.render(n, True, BLACK_COLOR)
        screen.blit(label, (rect.centerx - label.get_width() // 2, rect.bottom - 40))

#main app
def run_trombone_app():
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Magical Hand Trombone")
    title_font = pygame.font.Font(None, 72)
    info_font = pygame.font.Font(None, 32)

    #mediapipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    #webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    #note mapping
    note_names = ["Bb2", "C3", "D3", "F3", "Bb3"]
    note_freqs = [116.54, 130.81, 146.83, 174.61, 233.08]

    base_sounds = [generate_trombone_wave(freq, duration=1.2, amplitude=0.65, slide_cent=0) for freq in note_freqs]

    #state
    stable_fingers = {'Left': 0, 'Right': 0}
    pending_fingers = {'Left': 0, 'Right': 0}
    debounce_counters = {'Left': 0, 'Right': 0}

    #portamento
    current_play = {'channel': None, 'start_time': 0, 'freq': None}

    global PARTICLES

    clock = pygame.time.Clock()
    running = True
    while running:
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
        active_note = None

        slide_control = None

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handedness.classification[0].label
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                raw_fingers[hand_label] = count_fingers(hand_landmarks, hand_label)
                #thumb-index pixel distance
                try:
                    thumb = hand_landmarks.landmark[4]
                    index = hand_landmarks.landmark[8]
                    #x-distance
                    frame_h, frame_w, _ = frame.shape
                    dx = (thumb.x - index.x) * frame_w
                    dy = (thumb.y - index.y) * frame_h
                    dist = math.hypot(dx, dy)
                    slide_control = dist
                except:
                    pass

        #debounce/trigger
        for hand_label in ['Left', 'Right']:
            if raw_fingers[hand_label] == pending_fingers[hand_label]:
                debounce_counters[hand_label] += 1
            else:
                pending_fingers[hand_label] = raw_fingers[hand_label]
                debounce_counters[hand_label] = 0

            if debounce_counters[hand_label] >= DEBOUNCE_FRAMES:
                if stable_fingers[hand_label] != pending_fingers[hand_label]:
                    stable_fingers[hand_label] = pending_fingers[hand_label]
                    # trigger when fingers > 0
                    if stable_fingers[hand_label] > 0:
                        note_idx = stable_fingers[hand_label] - 1
                        if 0 <= note_idx < len(note_freqs):
                            target_freq = note_freqs[note_idx]
                            active_note = note_names[note_idx]
                            slide_cent = 0
                            if slide_control is not None:
                                sc = max(0.0, min(220.0, slide_control))
                                slide_cent = int((sc - 60.0) / 160.0 * 300.0)
                                slide_cent = max(-300, min(300, slide_cent))
                            try:
                                blip = generate_trombone_wave(target_freq, duration=0.18, amplitude=0.7, slide_cent=slide_cent)
                                blip.play()
                                base_snd = base_sounds[note_idx]
                                base_snd.play(-1)
                                current_play['channel'] = base_snd
                                current_play['start_time'] = time.time()
                                current_play['freq'] = target_freq
                            except Exception as e:
                                pass

                            pad_x = 100 + (SCREEN_WIDTH - 200) / len(note_names) * note_idx + ((SCREEN_WIDTH - 200) / len(note_names)) / 2
                            for _ in range(28):
                                PARTICLES.append(Particle(pad_x, SCREEN_HEIGHT - 120, KEY_PRESS_COLORS[note_idx % len(KEY_PRESS_COLORS)]))

        if stable_fingers['Left'] == 0 and stable_fingers['Right'] == 0:
            for snd in base_sounds:
                try:
                    snd.stop()
                except:
                    pass
            current_play['channel'] = None
            current_play['freq'] = None
            active_note = None
        else:
            for hand_label in ['Left', 'Right']:
                if stable_fingers[hand_label] > 0:
                    idx = stable_fingers[hand_label] - 1
                    if 0 <= idx < len(note_names):
                        active_note = note_names[idx]

        #drawing
        draw_background(screen)
        draw_trombone_ui(screen, note_names, active_note)

        #particles
        for p in PARTICLES:
            p.update()
        PARTICLES = [p for p in PARTICLES if p.lifespan > 0]
        for p in PARTICLES:
            p.draw(screen)

        draw_circular_webcam(screen, frame)

        title_text = title_font.render("Magical Hand Trombone", True, WHITE_COLOR)
        info_l = info_font.render(f"Left: {stable_fingers['Left']}", True, WHITE_COLOR)
        info_r = info_font.render(f"Right: {stable_fingers['Right']}", True, WHITE_COLOR)
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 40))
        screen.blit(info_l, (100, 60))
        screen.blit(info_r, (SCREEN_WIDTH - 100 - info_r.get_width(), 60))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    print("Application closed successfully.")

if __name__ == '__main__':
    run_trombone_app()
