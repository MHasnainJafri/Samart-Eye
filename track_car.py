import os
import sys
import platform
import json
import time
from threading import Thread
import threading
from datetime import datetime
import flet as ft

# List of car models
CAR_MODELS = [
    "vehicle_carlamotors_carlacola",
    "vehicle_dodge_charger_police",
    "vehicle_ford_crown",
    "vehicle_ford_mustang",
    "vehicle_harley-davidson_low_rider",
    "vehicle_jeep_wrangler_rubicon",
    "vehicle_kawasaki_ninja",
    "vehicle_lincoln_mkz_2020",
    "vehicle_mercedes_coupe_2020",
    "vehicle_micro_microlino",
    "vehicle_nissan_micra",
    "vehicle_nissan_patrol",
    "vehicle_tesla_cybertruck",
    "vehicle_volkswagen_t2"
]

def main(page: ft.Page):
    page.title = "Smart Eye"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_width = 1200
    page.window_height = 800
    page.window_resizable = True
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 30
    page.fonts = {
        "Roboto": "https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap"
    }
    page.theme = ft.Theme(font_family="Roboto")

    # State variables
    selected_car = ft.Dropdown(
        label="Select Car Model to Detect",
        options=[ft.dropdown.Option(model) for model in CAR_MODELS],
        width=300,
        autofocus=True,
        text_size=14,
        border_color=ft.Colors.BLUE_GREY_400
    )
    
    status_text = ft.Text("Ready to launch detection", size=14, color=ft.Colors.GREY_600)
    progress_ring = ft.ProgressRing(width=20, height=20, visible=False, stroke_width=2)
    
    # Image gallery with probability badges
    image_gallery = ft.GridView(
        expand=True,
        runs_count=4,
        max_extent=200,
        child_aspect_ratio=1.0,
        spacing=15,
        run_spacing=15,
    )
    
    # Detection timeline - now scrollable
    timeline = ft.ListView(
        spacing=10,
        auto_scroll=False,
        expand=True,
        height=300,
    )
    
    # Current selection info
    current_selection = ft.Column([
        ft.Text("No selection", size=14, color=ft.Colors.GREY_600),
        ft.Divider(height=5, color=ft.Colors.TRANSPARENT),
        ft.Row([
            ft.Icon(ft.Icons.CAMERA_ALT, size=16, color=ft.Colors.BLUE_700),
            ft.Text("", size=14, weight=ft.FontWeight.BOLD)
        ]),
        ft.Row([
            ft.Icon(ft.Icons.TIMER, size=16, color=ft.Colors.BLUE_700),
            ft.Text("", size=14)
        ]),
        ft.Row([
            ft.Icon(ft.Icons.STAR, size=16, color=ft.Colors.BLUE_700),
            ft.Text("", size=14)
        ])
    ], spacing=3)
    
    # Current car location section
    current_location = ft.Container(
        content=ft.Column([
            ft.Text("Last Known Location", size=16, weight=ft.FontWeight.BOLD),
            ft.Divider(height=10),
            ft.Row([
                ft.Icon(ft.Icons.LOCATION_ON, size=20, color=ft.Colors.RED_700),
                ft.Text("No recent detections", size=14, color=ft.Colors.GREY_600)
            ]),
            ft.Text("", size=12, color=ft.Colors.GREY_600),
            ft.Text("", size=12, color=ft.Colors.GREY_600)
        ], spacing=5),
        padding=15,
        border_radius=10,
        bgcolor=ft.Colors.WHITE,
        border=ft.border.all(1, ft.Colors.GREY_200),
        width=300
    )
    
    # Statistics cards
    def create_stat_card(title, value, icon, color):
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(icon, size=20, color=color),
                    ft.Text(title, size=14, color=ft.Colors.GREY_700)
                ]),
                ft.Text(value, size=24, weight=ft.FontWeight.BOLD)
            ], spacing=5),
            padding=15,
            border_radius=10,
            bgcolor=ft.Colors.WHITE,
            border=ft.border.all(1, ft.Colors.GREY_200),
            width=180,
            height=100
        )
    
    stats_row = ft.Row([
        create_stat_card("Total Detections", "0", ft.Icons.SEARCH, ft.Colors.BLUE_700),
        create_stat_card("Last Detection", "Never", ft.Icons.ACCESS_TIME, ft.Colors.GREEN_700),
        create_stat_card("Highest Conf.", "0%", ft.Icons.STAR, ft.Colors.AMBER_700),
        create_stat_card("Avg. Conf.", "0%", ft.Icons.TRENDING_UP, ft.Colors.PURPLE_700)
    ], spacing=15)
    
    # Confidence distribution chart
    def create_chart(data=None):
        if not data or len(data) == 0:
            return ft.Container(
                content=ft.Column([
                    ft.Icon(ft.Icons.BAR_CHART, size=50, color=ft.Colors.BLUE_GREY_300),
                    ft.Text("No detection data yet", size=16, color=ft.Colors.BLUE_GREY_400)
                ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                width=400,
                height=250,
                alignment=ft.alignment.center,
                bgcolor=ft.Colors.BLUE_GREY_50,
                border_radius=10,
            )
        
        # Create bins for the histogram
        bins = [0] * 10  # 0-10%, 10-20%, ..., 90-100%
        for detection in data:
            prob = detection.get("probability", 0)
            bin_index = min(int(prob * 10), 9)
            bins[bin_index] += 1
        
        max_count = max(bins) if max(bins) > 0 else 1
        
        chart_bars = []
        for i, count in enumerate(bins):
            height = (count / max_count) * 150
            chart_bars.append(
                ft.Column([
                    ft.Container(
                        width=30,
                        height=height,
                        bgcolor=ft.Colors.BLUE_400,
                        border_radius=ft.border_radius.only(top_left=5, top_right=5),
                    ),
                    ft.Text(f"{i*10}-{(i+1)*10}%", size=10)
                ], spacing=5, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
            )
        
        return ft.Container(
            content=ft.Column([
                ft.Text("Confidence Distribution", size=16, weight=ft.FontWeight.BOLD),
                ft.Row(
                    chart_bars,
                    alignment=ft.MainAxisAlignment.SPACE_EVENLY,
                    vertical_alignment=ft.CrossAxisAlignment.END,
                    height=200,
                ),
                ft.Text(f"Total samples: {len(data)}", size=12, color=ft.Colors.GREY_600)
            ], spacing=10),
            width=400,
            height=250,
            padding=15,
            bgcolor=ft.Colors.WHITE,
            border=ft.border.all(1, ft.Colors.GREY_200),
            border_radius=10,
        )

    chart_container = create_chart()

    def format_timestamp(timestamp_str):
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.strftime("%m/%d %H:%M:%S")
        except:
            return timestamp_str

    def update_detection_display(data):
        timeline.controls.clear()
        total_detections = len(data)
        last_detection = "Never"
        highest_confidence = 0
        avg_confidence = 0
        
        if data:
            last_detection = format_timestamp(data[-1]["timestamp"])
            confidences = [d.get("probability", 0) for d in data]
            highest_confidence = max(confidences) * 100
            avg_confidence = (sum(confidences) / len(confidences)) * 100
            
            # Update current location
            last_detection_data = data[-1]
            current_location.content.controls[2].controls[1].value = last_detection_data.get("camera_label", "Unknown location")
            current_location.content.controls[3].value = f"Time: {format_timestamp(last_detection_data.get('timestamp', ''))}"
            current_location.content.controls[4].value = f"Confidence: {last_detection_data.get('probability', 0)*100:.1f}%"
            
            for idx, detection in enumerate(data):
                camera = detection.get("camera_label", "Unknown")
                car_model = detection.get("predicted_class", "Unknown")
                prob = detection.get("probability", 0)
                timestamp = format_timestamp(detection.get("timestamp", ""))
                image_count = len(detection.get("crop_image", []))
                
                # Create a colored badge for confidence
                confidence_badge = ft.Container(
                    content=ft.Text(f"{prob*100:.0f}%", size=12, color=ft.Colors.WHITE),
                    bgcolor=ft.Colors.GREEN_600 if prob > 0.8 else ft.Colors.ORANGE_600 if prob > 0.5 else ft.Colors.RED_600,
                    padding=ft.padding.symmetric(horizontal=8, vertical=3),
                    border_radius=50
                )
                
                timeline_item = ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.LOCATION_ON, size=18, color=ft.Colors.BLUE_700),
                            ft.Text(camera, weight=ft.FontWeight.BOLD, size=14),
                            confidence_badge,
                            ft.Text(f"{image_count} images", size=12, color=ft.Colors.GREY_600)
                        ], spacing=10),
                        ft.Text(timestamp, size=12, color=ft.Colors.GREY_600),
                        ft.Text(car_model, size=12)
                    ], spacing=5),
                    padding=12,
                    border=ft.border.all(1, ft.Colors.GREY_200),
                    border_radius=8,
                    on_click=lambda e, d=detection: show_detection_images(d),
                    data=detection,
                    tooltip="Click to view images",
                    bgcolor=ft.Colors.WHITE,
                )
                timeline.controls.append(timeline_item)
        
        # Update statistics
        stats_row.controls[0].content.controls[1].value = str(total_detections)
        stats_row.controls[1].content.controls[1].value = last_detection
        stats_row.controls[2].content.controls[1].value = f"{highest_confidence:.0f}%"
        stats_row.controls[3].content.controls[1].value = f"{avg_confidence:.0f}%"
        
        # Update chart
        chart_container.content = create_chart(data).content
        
        page.update()

    def show_detection_images(detection):
        image_gallery.controls.clear()
        
        # Update selection info
        current_selection.controls[0].value = detection.get("predicted_class", "Unknown vehicle")
        current_selection.controls[2].controls[1].value = f"Camera {detection.get('camera_label', 'Unknown')}"
        current_selection.controls[3].controls[1].value = format_timestamp(detection.get("timestamp", ""))
        current_selection.controls[4].controls[1].value = f"{detection.get('probability', 0)*100:.1f}% confidence"
        
        for img_data in detection.get("crop_image", []):
            img_path = img_data.get("image_src", "")
            img_prob = img_data.get("probability", 0)
            
            # Handle path separators for different OS
            img_path = img_path.replace("\\", os.sep).replace("/", os.sep)
            abs_path = os.path.join(os.path.dirname(__file__), img_path)
            
            if os.path.exists(abs_path):
                # Create confidence badge
                prob_badge = ft.Container(
                    content=ft.Text(f"{img_prob*100:.0f}%", size=10, color=ft.Colors.WHITE),
                    bgcolor=ft.Colors.GREEN_600 if img_prob > 0.8 else ft.Colors.ORANGE_600 if img_prob > 0.5 else ft.Colors.RED_600,
                    padding=ft.padding.symmetric(horizontal=6, vertical=2),
                    border_radius=50,
                    top=5,
                    right=5,
                )
                
                # Create image container
                img_container = ft.Stack([
                    ft.Image(
                        src=abs_path,
                        fit=ft.ImageFit.COVER,
                        width=200,
                        height=200,
                        border_radius=ft.border_radius.all(8)
                    ),
                    prob_badge
                ], width=200, height=200)
                
                image_gallery.controls.append(
                    ft.Container(
                        content=img_container,
                        border_radius=8,
                        padding=0,
                        on_click=lambda e, path=abs_path: page.launch_url(path),
                        tooltip=f"Click to open\nConfidence: {img_prob*100:.1f}%",
                    )
                )
        
        page.update()

    def read_json_file(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def update_path_display():
        json_file_path = os.path.join(os.path.dirname(__file__), "detection/vehicle_detections.json")
        last_modified = 0
        
        while True:
            try:
                if os.path.exists(json_file_path):
                    current_modified = os.path.getmtime(json_file_path)
                    if current_modified != last_modified:
                        data = read_json_file(json_file_path)
                        update_detection_display(data)
                        last_modified = current_modified
            except Exception as e:
                status_text.value = f"Error reading JSON: {str(e)}"
                status_text.color = ft.Colors.RED
                page.update()
            
            time.sleep(1)

    def run_in_separate_terminal():
        if not selected_car.value:
            status_text.value = "Please select a car model."
            status_text.color = ft.Colors.RED
            page.update()
            return

        script_path = os.path.join(os.path.dirname(__file__), "self_model.py")

        if not os.path.exists(script_path):
            status_text.value = "Error: self_model.py not found!"
            status_text.color = ft.Colors.RED
            progress_ring.visible = False
            page.update()
            return

        try:
            status_text.value = "Launching detection in new terminal..."
            status_text.color = ft.Colors.BLUE_700
            progress_ring.visible = True
            page.update()

            car_model = selected_car.value

            if platform.system() == "Windows":
                command = f'start cmd /k "python {script_path} --car_to_detect {car_model}"'
            elif platform.system() == "Darwin":
                command = f'osascript -e \'tell application "Terminal" to do script "python {script_path} --car_to_detect {car_model}"\''
            else:
                command = f'x-terminal-emulator -e "python {script_path} --car_to_detect {car_model}"'

            def execute_command():
                os.system(command)
                status_text.value = "Detection running!"
                status_text.color = ft.Colors.GREEN_700
                progress_ring.visible = False
                page.update()

            Thread(target=execute_command, daemon=True).start()

        except Exception as e:
            status_text.value = f"Error: {str(e)}"
            status_text.color = ft.Colors.RED
            progress_ring.visible = False
            page.update()

    # Create the launch button with better styling
    launch_button = ft.ElevatedButton(
        text="Start Detection",
        icon=ft.Icons.PLAY_CIRCLE_FILL_OUTLINED,
        width=200,
        height=45,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
            padding=ft.padding.symmetric(horizontal=20, vertical=10),
            bgcolor=ft.Colors.BLUE_600,
            color=ft.Colors.WHITE,
            overlay_color=ft.Colors.BLUE_700,
            elevation=2,
        ),
        on_click=lambda e: run_in_separate_terminal()
    )

    # Start the JSON reading thread
    Thread(target=update_path_display, daemon=True).start()

    # Layout the page
    page.add(
        ft.Column([
            # Header
            ft.Row([
                ft.Icon(ft.Icons.DIRECTIONS_CAR_FILLED, size=36, color=ft.Colors.BLUE_700),
                ft.Text("Car Detection Analytics", size=28, weight=ft.FontWeight.BOLD),
            ], alignment=ft.MainAxisAlignment.CENTER),
            
            ft.Divider(height=20, color=ft.Colors.TRANSPARENT),
            
            # Controls row
            ft.Row([
                selected_car,
                launch_button,
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=30),
            
            ft.Row([progress_ring, status_text], alignment=ft.MainAxisAlignment.CENTER),
            
            ft.Divider(height=20),
            
            # Stats row
            stats_row,
            
            ft.Divider(height=20),
            
            # Main content
            ft.Row([
                # Left column - Timeline and location
                ft.Column([
                    ft.Text("Detection Timeline", size=18, weight=ft.FontWeight.BOLD),
                    ft.Container(
                        content=timeline,
                        height=300,
                        width=400,
                        border=ft.border.all(1, ft.Colors.GREY_200),
                        border_radius=10,
                        padding=10,
                        bgcolor=ft.Colors.WHITE
                    ),
                    ft.Divider(height=15),
                    current_location
                ], width=400, spacing=15),
                
                # Middle column - Images and details
                ft.Column([
                    current_selection,
                    ft.Container(
                        content=image_gallery,
                        height=400,
                        border=ft.border.all(1, ft.Colors.GREY_200),
                        border_radius=10,
                        padding=15,
                        bgcolor=ft.Colors.WHITE
                    ),
                    ft.Text("Click on timeline items to view detection images", size=12, color=ft.Colors.GREY_600, italic=True)
                ], expand=True, spacing=15),
                
                # Right sidebar - Chart and info
                ft.Column([
                    chart_container,
                    ft.Divider(height=15),
                    ft.Container(
                        content=ft.Column([
                            ft.Text("Detection Details", size=16, weight=ft.FontWeight.BOLD),
                            ft.Divider(height=10),
                            ft.Text("Each image shows its individual confidence score", size=12),
                            ft.Text("Hover over images to see details", size=12),
                            ft.Divider(height=10),
                            ft.Text("Color Coding:", size=14, weight=ft.FontWeight.BOLD),
                            ft.Row([ft.Container(width=15, height=15, bgcolor=ft.Colors.GREEN_600, border_radius=3), ft.Text("High confidence (>80%)", size=12)], spacing=5),
                            ft.Row([ft.Container(width=15, height=15, bgcolor=ft.Colors.ORANGE_600, border_radius=3), ft.Text("Medium confidence (50-80%)", size=12)], spacing=5),
                            ft.Row([ft.Container(width=15, height=15, bgcolor=ft.Colors.RED_600, border_radius=3), ft.Text("Low confidence (<50%)", size=12)], spacing=5)
                        ], spacing=5),
                        padding=15,
                        border_radius=10,
                        bgcolor=ft.Colors.WHITE,
                        border=ft.border.all(1, ft.Colors.GREY_200)
                    )
                ], width=300, spacing=15)
            ], spacing=30, expand=True)
        ], spacing=10, expand=True)
    )

if __name__ == "__main__":
    ft.app(target=main)