#!/usr/bin/env python3
"""
Interactive GUI for labeling player height in Geometry Dash dataset.

Controls:
  Left/Right arrows: Navigate between samples
  Click on image: Label player height (horizontal line + circle)
  Up arrow: Toggle "player not in frame" (shows X)
  Down arrow: Toggle "remove sample" (greyscale + darker)
  Q: Quit and save labels

Labels are saved to labels.json in the dataset folder.

Written using Claude Code
"""

import cv2
import numpy as np
import json
from pathlib import Path
import sys
from datetime import datetime


class DatasetLabeler:
    def __init__(self, dataset_path):
        self.dataset_dir = Path(dataset_path)
        self.metadata_path = self.dataset_dir / "metadata.json"
        session_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.labels_path = self.dataset_dir / f"labels_{session_ts}.json"

        if not self.metadata_path.exists():
            print(f"Error: {self.metadata_path} not found")
            sys.exit(1)

        with open(self.metadata_path) as f:
            self.metadata = json.load(f)

        self.num_samples = len(self.metadata)
        self.prior_labels = self.load_prior_labels()
        self.current_idx = self.find_start_idx()
        self.labels = {}

        self.window_name = "Dataset Labeler - Arrow keys to navigate | Click to label height | Up=not in frame | Down=remove | Q=quit"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 900, 900)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

    def load_prior_labels(self):
        """Merge all existing session files (sorted by timestamp). Later sessions win."""
        prior = {}
        label_files = sorted(self.dataset_dir.glob("labels_*.json"))
        for f in label_files:
            if f == self.labels_path:
                continue
            try:
                with open(f) as fp:
                    prior.update(json.load(fp))
            except Exception:
                pass
        print(f"  Loaded {len(prior)} prior labels from {len(label_files)} session file(s)")
        return prior

    def find_start_idx(self):
        """Return the index after the last labeled sample across all prior sessions."""
        labeled_ids = []
        for sid, lbl in self.prior_labels.items():
            if lbl.get("height_y") is not None or lbl.get("not_in_frame"):
                try:
                    labeled_ids.append(int(sid))
                except ValueError:
                    pass
        if not labeled_ids:
            return 0
        start = min(max(labeled_ids) + 1, self.num_samples - 1)
        print(f"  Resuming from sample {start} (last labeled: {max(labeled_ids)})")
        return start

    def get_effective_label(self, sample_id):
        """Current session label takes priority over prior sessions."""
        if sample_id in self.labels:
            return self.labels[sample_id], "current"
        if sample_id in self.prior_labels:
            return self.prior_labels[sample_id], "prior"
        return {}, None

    def save_labels(self):
        """Save labels to JSON file."""
        with open(self.labels_path, "w") as f:
            json.dump(self.labels, f, indent=2)
        print(f"✓ Labels saved to {self.labels_path}")

    def get_sample(self, idx):
        """Load a sample NPY file."""
        sample_file = self.dataset_dir / f"sample_{idx:04d}.npy"
        return np.load(sample_file)

    def on_mouse(self, event, x, y, flags, param):
        """Handle mouse clicks for height labeling."""
        if event == cv2.EVENT_LBUTTONDOWN:
            sample_id = str(self.metadata[self.current_idx]["sample_id"])

            # Get current label or create new
            if sample_id not in self.labels:
                self.labels[sample_id] = {"removed": False, "not_in_frame": False}

            # Set height label and clear not_in_frame
            self.labels[sample_id]["height_y"] = y
            self.labels[sample_id]["height_x"] = x
            self.labels[sample_id]["not_in_frame"] = False

            self.save_labels()  # Auto-save
            print(f"  Labeled height at y={y}, x={x}")

            # Move to next datapoint
            self.navigate("right")

    def display_current(self):
        """Display the current sample with annotations from most recent label."""
        sample = self.get_sample(self.current_idx)

        # Extract RGB channels (first 3 channels)
        img = sample[:, :, :3].copy()

        # Get label info — prefer current session, fall back to prior
        sample_id = str(self.metadata[self.current_idx]["sample_id"])
        label, label_source = self.get_effective_label(sample_id)

        # Apply greyscale + darker if removed
        if label.get("removed", False):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            img = (img * 0.5).astype(np.uint8)

        # Determine color: cyan for prior sessions, red for current session
        line_color = (255, 255, 0) if label_source == "prior" else (0, 0, 255)  # BGR: cyan or red

        # Draw annotations
        if label.get("not_in_frame", False):
            # Draw X (diagonal lines across entire image)
            cv2.line(img, (0, 0), (660, 660), line_color, 1)
            cv2.line(img, (660, 0), (0, 660), line_color, 1)
        elif "height_y" in label:
            # Draw horizontal line at height_y
            y = label["height_y"]
            x = label["height_x"]
            cv2.line(img, (0, y), (660, y), line_color, 1)
            # Draw circle at click position
            cv2.circle(img, (x, y), 4, line_color, -1)
            cv2.circle(img, (x, y), 4, (255, 255, 255), 1)

        # Add title with sample counter
        title = f"{self.current_idx + 1}/{self.num_samples}"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        title_x = (660 - title_size[0]) // 2
        cv2.putText(
            img,
            title,
            (title_x, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )

        # Add status information at bottom
        status_lines = []
        meta = self.metadata[self.current_idx]
        status_lines.append(f"Video: {Path(meta['video_path']).name} | Frame: {meta['frame_index']}")

        # Show label source (prior/current) and state
        if label_source == "prior":
            if label.get("not_in_frame"):
                status_lines.append("[prior] [X] Player NOT in frame")
            elif "height_y" in label:
                status_lines.append(f"[prior] [✓] Height: y={label['height_y']}, x={label['height_x']}")
            else:
                status_lines.append("[prior] [ ] (unlabeled in session)")
        elif label_source == "current":
            if label.get("not_in_frame"):
                status_lines.append("[NEW] [X] Player NOT in frame")
            elif "height_y" in label:
                status_lines.append(f"[NEW] [✓] Height: y={label['height_y']}, x={label['height_x']}")
            else:
                status_lines.append("[NEW] [ ] (labeled in this session)")
        else:
            status_lines.append("[ ] Click to label height")

        if label.get("removed"):
            status_lines.append("[REMOVED - greyscale]")

        y_offset = 630
        for status in status_lines:
            cv2.putText(img, status, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            y_offset -= 25

        cv2.imshow(self.window_name, img)

    def navigate(self, direction):
        """Navigate to next/previous sample."""
        if direction == "left":
            self.current_idx = max(0, self.current_idx - 1)
        elif direction == "right":
            self.current_idx = min(self.num_samples - 1, self.current_idx + 1)
        self.display_current()

    def toggle_not_in_frame(self):
        """Toggle 'player not in frame' label."""
        sample_id = str(self.metadata[self.current_idx]["sample_id"])
        if sample_id not in self.labels:
            self.labels[sample_id] = {}

        # Get effective current state (consider prior labels)
        effective_label, _ = self.get_effective_label(sample_id)
        current = effective_label.get("not_in_frame", False)

        # Toggle in current session
        self.labels[sample_id]["not_in_frame"] = not current

        # Clear height label when marking as not in frame
        if self.labels[sample_id]["not_in_frame"]:
            self.labels[sample_id].pop("height_y", None)
            self.labels[sample_id].pop("height_x", None)

        self.save_labels()  # Auto-save
        self.display_current()
        state = "marked" if not current else "unmarked"
        print(f"  Player not in frame: {state}")

    def toggle_removed(self):
        """Toggle 'remove sample' label."""
        sample_id = str(self.metadata[self.current_idx]["sample_id"])
        if sample_id not in self.labels:
            self.labels[sample_id] = {}

        # Get effective current state (consider prior labels)
        effective_label, _ = self.get_effective_label(sample_id)
        current = effective_label.get("removed", False)

        # Toggle in current session
        self.labels[sample_id]["removed"] = not current

        self.save_labels()  # Auto-save
        self.display_current()
        state = "removed" if not current else "restored"
        print(f"  Sample {state}")

    def process_key(self, key):
        """Process keyboard input."""
        # Handle different possible key codes for arrow keys
        # (varies by platform and OpenCV version)

        # Regular ASCII keys
        if key == ord("q") or key == ord("Q"):
            return "quit"

        # Arrow keys - try multiple possible codes
        # Linux/Windows typically use: 81-84
        # macOS may use different codes
        if key in [81, 1111164]:  # Left arrow
            self.navigate("left")
        elif key in [83, 1111166]:  # Right arrow
            self.navigate("right")
        elif key in [82, 1111163]:  # Up arrow
            self.toggle_not_in_frame()
        elif key in [84, 1111165]:  # Down arrow
            self.toggle_removed()

        # WASD alternative controls
        elif key == ord("a") or key == ord("A"):  # A = left
            self.navigate("left")
        elif key == ord("d") or key == ord("D"):  # D = right
            self.navigate("right")
        elif key == ord("w") or key == ord("W"):  # W = up
            self.toggle_not_in_frame()
        elif key == ord("s") or key == ord("S"):  # S = down
            self.toggle_removed()

        elif key == -1:
            pass  # No key pressed

    def run(self):
        """Main event loop."""
        print(f"Loaded {self.num_samples} samples from {self.dataset_dir}/")
        print("\n" + "="*60)
        print("DATASET LABELER - CONTROLS")
        print("="*60)
        print("\nNAVIGATION:")
        print("  Left Arrow / A  : Previous sample")
        print("  Right Arrow / D : Next sample")
        print("\nLABELING:")
        print("  Click on image  : Mark player height with red line + circle")
        print("  Up Arrow / W    : Toggle 'player NOT in frame' (shows X)")
        print("  Down Arrow / S  : Toggle 'remove from training' (greyscale)")
        print("\nGENERAL:")
        print("  Q               : Quit and save all labels")
        print("  X button        : Close window and save all labels")
        print("\n" + "="*60)
        print("• All changes are auto-saved after each action")
        print("• Safe to close anytime - no data loss")
        print(f"• Labels saved to: {self.labels_path.name}")
        print("="*60 + "\n")

        self.display_current()

        while True:
            # Use small timeout (1ms) to keep window responsive on macOS
            key = cv2.waitKey(1)

            # Check if window was closed via X button
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            # Process key only if one was pressed (waitKey returns -1 if no key)
            if key != -1:
                action = self.process_key(key)
                if action == "quit":
                    break

        self.save_labels()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset_5_0"

    if not Path(dataset_dir).exists():
        print(f"Error: Dataset directory '{dataset_dir}' not found")
        sys.exit(1)

    try:
        labeler = DatasetLabeler(dataset_dir)
        labeler.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving labels...")
        labeler.save_labels()
        print("Done.")
        sys.exit(0)
