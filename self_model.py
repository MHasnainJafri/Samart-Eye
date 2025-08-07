import carla
import pygame
import numpy as np
import json
import math
import cv2
import time
import os
from collections import deque
from queue import Queue
from threading import Thread
import logging
from ultralytics import YOLO
import torch
from torchvision import models, transforms
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.DEBUG, filename='carla_app.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Pygame
pygame.init()
grid_width, grid_height = 1200, 800
grid_display = pygame.display.set_mode((grid_width, grid_height))
pygame.display.set_caption("CARLA Camera Grid with ROI and T2 Tracking")

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Initialize YOLO model
yolo_model = None
try:
    yolo_model = YOLO('yolo11s.pt').to(device)
    logging.info(f"YOLO model loaded on {device}")
    _ = yolo_model.predict(np.zeros((640, 360, 3), dtype=np.uint8), verbose=False)
except Exception as e:
    logging.error(f"Failed to initialize YOLO model: {str(e)}")
    yolo_model = None

# Load car model classifier
def load_model(model_path, class_names, device):
    try:
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, len(class_names))
        )
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        logging.info(f"Loaded car model classifier from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading car model classifier: {str(e)}")
        raise

try:
    with open('../ModelTraining/car_model_classes.json', 'r') as f:
        class_names = json.load(f)
    logging.info(f"Loaded class names: {class_names}")
    classifier_model = load_model("../ModelTraining/checkpoints_20250725_183712/best_model.pth", class_names, device)
except Exception as e:
    logging.error(f"Failed to load classifier or class names: {str(e)}")
    classifier_model = None

classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_car_model(image, model, class_names, device):
    try:
        img = Image.fromarray(image).convert('RGB')
        img = classifier_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)[0]
            _, predicted = torch.max(outputs, 1)
            pred_class = class_names[predicted.item()]
            pred_prob = probs[predicted.item()].item()
            prob_dict = {class_names[i]: probs[i].item() for i in range(len(class_names))}
            logging.debug(f"Classifier probabilities: {prob_dict}")
        return pred_class, pred_prob
    except Exception as e:
        logging.error(f"Error predicting car model: {str(e)}")
        return "Unknown", 0.0

# Connect to CARLA
client = carla.Client('10.20.20.91', 2000)
client.set_timeout(15.0)
world = client.get_world()

# Set asynchronous mode
settings = world.get_settings()
settings.synchronous_mode = False
world.apply_settings(settings)

# Video recording parameters
RECORDING_WIDTH = 800
RECORDING_HEIGHT = 600
FPS = 20
DETECTION_THRESHOLD = 0.5
MIN_ROI_SIZE = 10

# Spawn vehicles in CARLA
def spawn_vehicles(world, num_vehicles=10):
    blueprint_library = world.get_blueprint_library()
    vehicle_bps = blueprint_library.filter('vehicle.*')
    # Ensure at least one vehicle_volkswagen_t2 is spawned
    t2_bp = next((bp for bp in vehicle_bps if 'vehicle.vehicle_dodge_charger_police' in bp.id), vehicle_bps[0])
    spawn_points = world.get_map().get_spawn_points()
    for i in range(min(num_vehicles, len(spawn_points))):
        bp = t2_bp if i == 0 else vehicle_bps[i % len(vehicle_bps)]
        try:
            world.spawn_actor(bp, spawn_points[i])
            logging.info(f"Spawned vehicle {i+1}: {bp.id}")
        except Exception as e:
            logging.error(f"Failed to spawn vehicle {i+1}: {str(e)}")

def save_rois(street_renders, filename='camera_rois.json'):
    try:
        roi_data = []
        for render in street_renders:
            if render.roi is not None:
                roi_data.append({"label": render.label, "roi": render.roi})
        with open(filename, 'w') as f:
            json.dump(roi_data, f, indent=4)
        logging.info(f"Saved ROIs to {filename}")
    except Exception as e:
        logging.error(f"Error saving ROIs to {filename}: {str(e)}")

def load_rois(street_renders, filename='camera_rois.json'):
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                roi_data = json.load(f)
            for roi_entry in roi_data:
                label = roi_entry.get("label")
                roi = roi_entry.get("roi")
                if isinstance(roi, (list, tuple)) and len(roi) == 4:
                    for render in street_renders:
                        if render.label == label:
                            x1, y1, x2, y2 = roi
                            if abs(x2 - x1) >= MIN_ROI_SIZE and abs(y2 - y1) >= MIN_ROI_SIZE:
                                render.roi = tuple(roi)
                                logging.info(f"Loaded ROI for {label}: {render.roi}")
                            else:
                                logging.warning(f"Invalid ROI for {label}: {roi} (too small)")
                            break
        else:
            logging.info(f"No ROI file found at {filename}")
    except Exception as e:
        logging.error(f"Error loading ROIs from {filename}: {str(e)}")

class CameraRenderObject:
    def __init__(self, width, height, label):
        self.width = width
        self.height = height
        self.label = label
        self.surface = pygame.Surface((width, height))
        self.font = pygame.font.SysFont('Arial', 16)
        self.last_frame = None
        self.vehicle_boxes = []
        self.vehicle_labels = []
        self.fps_queue = deque(maxlen=10)
        self.detection_fps_queue = deque(maxlen=10)
        self.last_time = time.time()
        self.roi = None
        self.drawing_roi = False
        self.roi_start = None
        self.frame_queue = Queue(maxsize=1)
        self.t2_detected = False  # Track if vehicle_volkswagen_t2 is detected
        self.thread = Thread(target=self.process_frames, daemon=True)
        self.thread.start()

    def process_frames(self):
        while True:
            image = self.frame_queue.get()
            if image is None:
                break
            self.update(image)

    def update(self, image):
        try:
            start_time = time.time()
            img = np.frombuffer(image.raw_data, dtype=np.uint8)
            img = img.reshape((image.height, image.width, 4))
            img = img[:, :, :3][:, :, ::-1]  # BGR to RGB
            self.last_frame = img
            
            if self.width != image.width or self.height != image.height:
                img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            self.surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            
            self.fps_queue.append(1/(time.time() - start_time + 1e-6))
            logging.debug(f"Frame processed for {self.label}, time: {(time.time() - start_time)*1000:.2f} ms")
        except Exception as e:
            logging.error(f"Error updating camera frame for {self.label}: {str(e)}")

    def detect_vehicles(self, results, idx):
        if self.last_frame is None or yolo_model is None or classifier_model is None:
            logging.warning(f"Skipping detection for {self.label}: No frame or models")
            return
        try:
            start_time = time.time()
            img = self.last_frame.copy()
            self.t2_detected = False  # Reset detection flag
            if self.roi is not None:
                x1, y1, x2, y2 = self.roi
                x1_orig = max(0, int(x1 * self.last_frame.shape[1] / self.width))
                y1_orig = max(0, int(y1 * self.last_frame.shape[0] / self.height))
                x2_orig = min(self.last_frame.shape[1], int(x2 * self.last_frame.shape[1] / self.width))
                y2_orig = min(self.last_frame.shape[0], int(y2 * self.last_frame.shape[0] / self.height))
                if x2_orig <= x1_orig or y2_orig <= y1_orig:
                    logging.warning(f"Invalid ROI for {self.label}: {self.roi}")
                    return
                img = img[y1_orig:y2_orig, x1_orig:x2_orig]
                if img.size == 0:
                    logging.warning(f"Empty ROI crop for {self.label}")
                    return

            self.vehicle_boxes = []
            self.vehicle_labels = []
            scale_x = self.width / self.last_frame.shape[1]
            scale_y = self.height / self.last_frame.shape[0]

            for box in results[idx].boxes:
                if box.conf.item() > DETECTION_THRESHOLD:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Adjust coordinates for ROI offset
                    if self.roi is not None:
                        roi_x1, roi_y1, _, _ = self.roi
                        x1 += roi_x1 * self.last_frame.shape[1] / self.width
                        x2 += roi_x1 * self.last_frame.shape[1] / self.width
                        y1 += roi_y1 * self.last_frame.shape[0] / self.height
                        y2 += roi_y1 * self.last_frame.shape[0] / self.height
                        # Check if box is within ROI
                        if not (self.roi[0] <= x1 * scale_x <= self.roi[2] and
                                self.roi[0] <= x2 * scale_x <= self.roi[2] and
                                self.roi[1] <= y1 * scale_y <= self.roi[3] and
                                self.roi[1] <= y2 * scale_y <= self.roi[3]):
                            continue
                    x1_orig = max(0, int(x1))
                    y1_orig = max(0, int(y1))
                    x2_orig = min(self.last_frame.shape[1], int(x2))
                    y2_orig = min(self.last_frame.shape[0], int(y2))
                    if x2_orig > x1_orig and y2_orig > y1_orig:
                        crop = self.last_frame[y1_orig:y2_orig, x1_orig:x2_orig]
                        if crop.size == 0:
                            logging.warning(f"Empty crop for {self.label} at ({x1_orig}, {y1_orig}, {x2_orig}, {y2_orig})")
                            continue
                        pred_class, pred_prob = predict_car_model(crop, classifier_model, class_names, device)
                        self.vehicle_boxes.append((int(x1 * scale_x), int(y1 * scale_y), int((x2 - x1) * scale_x), int((y2 - y1) * scale_y)))
                        self.vehicle_labels.append(f"{pred_class} ({pred_prob:.2f})")
                        if pred_class == "vehicle_dodge_charger_police" and pred_prob > DETECTION_THRESHOLD:
                            self.t2_detected = True
                        logging.debug(f"Detection for {self.label}: Class={pred_class}, Conf={pred_prob:.2f}, Box=({x1}, {y1}, {x2}, {y2})")
            self.detection_fps_queue.append(1/(time.time() - start_time + 1e-6))
            logging.info(f"Detected {len(self.vehicle_boxes)} vehicles for {self.label}, T2 detected: {self.t2_detected}")
        except Exception as e:
            logging.error(f"Detection error for {self.label}: {str(e)}")

def setup_cameras():
    json_filename = 'camera_transforms.json'
    try:
        with open(json_filename) as f:
            camera_data = json.load(f)
        logging.info(f"Loaded {len(camera_data)} camera positions")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"No camera positions found: {str(e)}")
        return [], []

    spawn_vehicles(world, num_vehicles=10)

    street_cameras = []
    street_renders = []
    grid_cols = 3
    grid_rows = 2
    cell_width = grid_width // grid_cols
    cell_height = grid_height // grid_rows

    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(RECORDING_WIDTH))
    camera_bp.set_attribute('image_size_y', str(RECORDING_HEIGHT))
    camera_bp.set_attribute('fov', '90')

    for i, cam_info in enumerate(camera_data):
        loc = cam_info["location"]
        rot = cam_info["rotation"]
        transform = carla.Transform(
            carla.Location(x=loc["x"], y=loc["y"], z=loc["z"]),
            carla.Rotation(pitch=rot["pitch"], yaw=rot["yaw"], roll=rot["roll"])
        )
        render = CameraRenderObject(cell_width, cell_height, f"Cam {i+1}")
        street_renders.append(render)
        try:
            camera = world.spawn_actor(camera_bp, transform)
            camera.listen(lambda image, idx=i: street_renders[idx].frame_queue.put(image))
            street_cameras.append(camera)
        except Exception as e:
            logging.error(f"Failed to spawn camera {i+1}: {str(e)}")

    load_rois(street_renders)
    return street_cameras, street_renders

def create_video_writers(base_path, num_cameras):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(base_path, exist_ok=True)
    writers = []
    for i in range(num_cameras):
        filename = os.path.join(base_path, f"camera_{i+1}_{timestamp}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(filename, fourcc, FPS, (RECORDING_WIDTH, RECORDING_HEIGHT), isColor=True)
        if not writer.isOpened():
            logging.error(f"Failed to initialize video writer for camera {i+1}")
            writer = None
        writers.append(writer)
    return writers

def draw_detections(surface, boxes, labels, fps, detection_fps, roi, drawing_roi, roi_start, grid_x, grid_y, t2_detected, frame_count):
    if roi is not None:
        pygame.draw.rect(surface, (0, 255, 0), (roi[0], roi[1], roi[2] - roi[0], roi[3] - roi[1]), 2)
    if drawing_roi and roi_start is not None:
        mouse_pos = pygame.mouse.get_pos()
        x, y = mouse_pos[0] - grid_x, mouse_pos[1] - grid_y
        x = max(0, min(x, surface.get_width()))
        y = max(0, min(y, surface.get_height()))
        pygame.draw.rect(surface, (0, 255, 0), (roi_start[0], roi_start[1], x - roi_start[0], y - roi_start[1]), 2)

    for box, label in zip(boxes, labels):
        x, y, w, h = box
        pygame.draw.rect(surface, (255, 0, 0), (x, y, w, h), 2)
        text_surface = pygame.font.SysFont('Arial', 16).render(label, True, (255, 255, 255))
        surface.blit(text_surface, (x, y - 20 if y > 20 else y + h + 5))

    # Draw blinking red border if vehicle_volkswagen_t2 is detected
    if t2_detected and (frame_count // 10) % 2 == 0:  # Blink every 10 frames
        pygame.draw.rect(surface, (255, 0, 0), (0, 0, surface.get_width(), surface.get_height()), 5)

    fps_text = f"Display: {fps:.1f} FPS"
    det_fps_text = f"Detection: {detection_fps:.1f} FPS"
    font = pygame.font.SysFont('Arial', 16)
    surface.blit(font.render(fps_text, True, (255, 255, 255)), (10, 10))
    surface.blit(font.render(det_fps_text, True, (255, 255, 255)), (10, 30))

def main():
    street_cameras, street_renders = setup_cameras()
    if not street_cameras:
        logging.error("No cameras loaded, exiting.")
        return

    clock = pygame.time.Clock()
    running = True
    recording = False
    video_writers = []
    grid_cols = 3
    grid_rows = 2
    detection_interval = 1.0
    selected_camera = None
    last_detection_time = time.time()
    render_counter = 0
    frame_count = 0
    os.makedirs("debug", exist_ok=True)

    try:
        while running:
            clock.tick(FPS)
            frame_count += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        recording = not recording
                        if recording:
                            logging.info("Starting recording...")
                            video_writers = create_video_writers("recordings", len(street_cameras))
                        else:
                            logging.info("Stopping recording...")
                            for writer in video_writers:
                                if writer is not None:
                                    writer.release()
                            video_writers = []
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_c:
                        if selected_camera is not None:
                            street_renders[selected_camera].roi = None
                            logging.info(f"Cleared ROI for {street_renders[selected_camera].label}")
                            save_rois(street_renders)
                            selected_camera = None
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_pos = event.pos
                        for i, render in enumerate(street_renders):
                            row = i // grid_cols
                            col = i % grid_cols
                            x_pos = col * render.width
                            y_pos = row * render.height
                            if x_pos <= mouse_pos[0] < x_pos + render.width and y_pos <= mouse_pos[1] < y_pos + render.height:
                                render.drawing_roi = True
                                render.roi_start = (mouse_pos[0] - x_pos, mouse_pos[1] - y_pos)
                                selected_camera = i
                                logging.info(f"Started drawing ROI for {render.label} at {render.roi_start}")
                                break
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and selected_camera is not None:
                        render = street_renders[selected_camera]
                        if render.drawing_roi:
                            render.drawing_roi = False
                            mouse_pos = event.pos
                            row = selected_camera // grid_cols
                            col = selected_camera % grid_cols
                            x_pos = col * render.width
                            y_pos = row * render.height
                            x1 = render.roi_start[0]
                            y1 = render.roi_start[1]
                            x2 = max(0, min(mouse_pos[0] - x_pos, render.width))
                            y2 = max(0, min(mouse_pos[1] - y_pos, render.height))
                            if abs(x2 - x1) >= MIN_ROI_SIZE and abs(y2 - y1) >= MIN_ROI_SIZE:
                                render.roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y2, y2))
                                logging.info(f"Set ROI for {render.label}: {render.roi}")
                                save_rois(street_renders)
                            else:
                                logging.warning(f"Invalid ROI for {render.label}: Too small")
                                render.roi = None
                            render.roi_start = None
                            selected_camera = None

            # Batch YOLO detection
            if time.time() - last_detection_time > detection_interval:
                images = []
                render_indices = []
                for i, render in enumerate(street_renders):
                    if render.last_frame is not None:
                        if render.roi is not None:
                            x1, y1, x2, y2 = render.roi
                            x1_orig = max(0, int(x1 * render.last_frame.shape[1] / render.width))
                            y1_orig = max(0, int(y1 * render.last_frame.shape[0] / render.height))
                            x2_orig = min(render.last_frame.shape[1], int(x2 * render.last_frame.shape[1] / render.width))
                            y2_orig = min(render.last_frame.shape[0], int(y2 * render.last_frame.shape[0] / render.height))
                            if x2_orig > x1_orig and y2_orig > y1_orig:
                                images.append(render.last_frame[y1_orig:y2_orig, x1_orig:x2_orig])
                                render_indices.append(i)
                        else:
                            images.append(render.last_frame)
                            render_indices.append(i)
                if images and yolo_model is not None:
                    results = yolo_model.predict(images, verbose=True, classes=[2, 3, 5, 7], device=device, imgsz=960, conf=DETECTION_THRESHOLD)
                    for res_idx, render_idx in enumerate(render_indices):
                        street_renders[render_idx].detect_vehicles(results, res_idx)
                        debug_img = results[res_idx].plot()
                        cv2.imwrite(f"debug/{street_renders[render_idx].label}_{time.strftime('%Y%m%d_%H%M%S')}.jpg", debug_img)
                    logging.info(f"YOLO detected {sum(len(r.boxes) for r in results)} objects across {len(images)} cameras")
                last_detection_time = time.time()

            render_counter += 1
            if render_counter % 2 == 0:
                grid_display.fill((0, 0, 0))
                for i, render in enumerate(street_renders):
                    surface = render.surface.copy()
                    avg_fps = sum(render.fps_queue)/len(render.fps_queue) if render.fps_queue else 0
                    avg_det_fps = sum(render.detection_fps_queue)/len(render.detection_fps_queue) if render.detection_fps_queue else 0
                    row = i // grid_cols
                    col = i % grid_cols
                    x_pos = col * render.width
                    y_pos = row * render.height
                    draw_detections(surface, render.vehicle_boxes, render.vehicle_labels, avg_fps, avg_det_fps, render.roi, render.drawing_roi, render.roi_start, x_pos, y_pos, render.t2_detected, frame_count)
                    
                    if recording and i < len(video_writers) and video_writers[i] is not None and render.last_frame is not None:
                        frame = render.last_frame.copy()
                        scale_x = frame.shape[1] / render.width
                        scale_y = frame.shape[0] / render.height
                        if render.roi is not None:
                            x1, y1, x2, y2 = render.roi
                            x1_orig = int(x1 * scale_x)
                            y1_orig = int(y1 * scale_y)
                            x2_orig = int(x2 * scale_x)
                            y2_orig = int(y2 * scale_y)
                            cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
                        for box, label in zip(render.vehicle_boxes, render.vehicle_labels):
                            x, y, w, h = box
                            x_orig = int(x * scale_x)
                            y_orig = int(y * scale_y)
                            w_orig = int(w * scale_x)
                            h_orig = int(h * scale_y)
                            cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (0, 0, 255), 2)
                            cv2.putText(frame, label, (x_orig, y_orig - 10 if y_orig > 10 else y_orig + h_orig + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        if render.t2_detected and (frame_count // 10) % 2 == 0:
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        video_writers[i].write(frame)
                    
                    grid_display.blit(surface, (x_pos, y_pos))
                    label = render.font.render(render.label, True, (255, 255, 255))
                    grid_display.blit(label, (x_pos + 10, y_pos + 10))
                
                pygame.display.flip()

    finally:
        logging.info("Cleaning up...")
        for render in street_renders:
            render.frame_queue.put(None)
        for camera in street_cameras:
            if camera.is_alive:
                try:
                    camera.destroy()
                except Exception as e:
                    logging.error(f"Error destroying camera: {str(e)}")
        for writer in video_writers:
            if writer is not None and writer.isOpened():
                try:
                    writer.release()
                except Exception as e:
                    logging.error(f"Error releasing video writer: {str(e)}")
        pygame.quit()

if __name__ == "__main__":
    main()