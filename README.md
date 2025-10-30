# Close Person Detection

This project detects nearby people in a video stream by combining **object detection** and **depth estimation**.  
It highlights individuals who are closer than a defined distance threshold.

---

## Possible utilities and aplications
- Last-response safety for self-driving cars — trigger emergency braking or evasive maneuvers when a person is detected too close to the vehicle.
- Personal assistant robot safety — dynamically adjust robot behavior based on human distance and presence.
- Smart home automation — trigger actions (e.g., lights, heating, or security cameras) when a person is detected within a certain range.
- Gaming interfaces — enable proximity-based player interactions or adaptive experiences.
 - Security surveillance — detect unauthorized proximity to restricted areas or valuable equipment.

## Demo Video
[demo.webm](https://github.com/user-attachments/assets/041c215a-568f-42db-915d-bff2ee476140)


## Project Structure

```
close_person_detection/
│
├── scripts/
│   ├── main.py
│   ├── viewer.py
│   ├── object_detection.py
│   ├── depth_estimation.py
│   └── close_person_estimation.py
│
└── media/
    └── video.mp4
```

---

## Features

- YOLO-based object detection for people.
- Depth-Anything-V2 depth estimation for distance calculation.
- Proximity analysis with alerts (messages and colors red/green border).
- Real-time visualization combining detection + depth map.

---

## Installation

Follow these steps to install all dependencies and set up the Docker environment:

### 1. Clone the repository
```bash
git clone https://github.com/ANadalCardenas/close_person_detection.git
cd close_person_detection
```

### 2. Build the Docker image
```bash
docker compose build close_person_detection
```

### 3. Run the Docker container
```bash
docker compose up close_person_detection
```

### 4. Access the container shell
```bash
docker exec -it close_person_detection_container bash
```

---

## ▶️ Usage

1. Place your input video in `close_person_detection/media/video.mp4`.
2. Run the script:

```bash
python scripts/main.py
```

3. Press **'q'** to quit the visualization window.

---

## ⚙️ Configuration

Set the depth limit in `main.py` (DEPTH_LIMIT):

```python

DEPTH_LIMIT = 0.01

```

---

## 📄 License

This project is released under the MIT License. [(License.txt)](License.txt)  
Feel free to use, modify, and distribute it for educational and research purposes.
