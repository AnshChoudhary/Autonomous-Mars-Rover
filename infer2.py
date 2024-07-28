import cv2
import numpy as np
import torch
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import Accuracy, ConfusionMatrix, JaccardIndex
from tqdm import tqdm
import logging

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.dc1 = double_conv(n_channels, 64)
        self.dc2 = double_conv(64, 128)
        self.dc3 = double_conv(128, 256)
        self.dc4 = double_conv(256, 512)
        self.dc5 = double_conv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.dc6 = double_conv(1024, 512)
        self.dc7 = double_conv(512, 256)
        self.dc8 = double_conv(256, 128)
        self.dc9 = double_conv(128, 64)

        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = self.dc1(x)
        x2 = self.dc2(nn.functional.max_pool2d(x1, 2))
        x3 = self.dc3(nn.functional.max_pool2d(x2, 2))
        x4 = self.dc4(nn.functional.max_pool2d(x3, 2))
        x5 = self.dc5(nn.functional.max_pool2d(x4, 2))

        x = self.up1(x5)
        x = self.dc6(torch.cat([x4, x], dim=1))
        x = self.up2(x)
        x = self.dc7(torch.cat([x3, x], dim=1))
        x = self.up3(x)
        x = self.dc8(torch.cat([x2, x], dim=1))
        x = self.up4(x)
        x = self.dc9(torch.cat([x1, x], dim=1))
        
        return self.final(x)

class ImageSegmentationModel(pl.LightningModule):
    def __init__(self, num_classes: int = 5, learning_rate: float = 1e-4):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.model = UNet(n_channels=3, n_classes=num_classes)

        self.loss = nn.CrossEntropyLoss()
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.iou = JaccardIndex(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        preds = torch.argmax(preds, dim=1)

        # Log metrics
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.accuracy(preds, y), on_step=True, on_epoch=True)
        self.log('val_iou', self.iou(preds, y), on_step=True, on_epoch=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Perform Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    # Create a blank mask to draw the lines
    line_mask = np.zeros_like(frame)
    
    # Draw the detected lines on the mask
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Combine the original frame with the line mask
    result = cv2.addWeighted(frame, 0.8, line_mask, 1, 0)
    
    return result

def process_frame(frame, model, device):
    # Preprocess the frame
    frame = cv2.resize(frame, (256, 256))
    frame = np.asarray(frame, dtype=np.float32) / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = torch.from_numpy(frame).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        prediction = model(frame)
    
    # Post-process the prediction
    segmentation_map = torch.argmax(prediction, dim=1).squeeze().cpu().numpy()
    
    return segmentation_map

def create_overlay(segmentation_map):
    # Define colors for different terrain types in AI4MARS dataset
    colors = [
        (0, 0, 0),      # Class 0: Background/Sky (Black)
        (255, 0, 0),    # Class 1: Sand (Red)
        (0, 255, 0),    # Class 2: Bedrock (Green)
        (0, 0, 255),    # Class 3: Big Rocks (Blue)
        (255, 255, 0)   # Class 4: Small Rocks (Yellow)
    ]
    
    colored_segmentation = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        colored_segmentation[segmentation_map == i] = color
    
    return colored_segmentation

def main():
    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # Load the trained model
    logging.info("Loading the trained model...")
    model = ImageSegmentationModel()
    model.load_state_dict(torch.load("trained_unet_model_checkpoint.pth", map_location=device))
    model.eval()
    model.to(device)
    logging.info("Model loaded successfully")
    
    # Open the input video
    input_video_path = '/Users/anshchoudhary/Downloads/AI4MARS/mars_input.mp4'
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {input_video_path}")
        return
    
    logging.info(f"Processing video: {input_video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logging.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Create output video writer
    output_video_path = '/Users/anshchoudhary/Downloads/AI4MARS/outputs/mars_output_hough.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Process frames with tqdm progress bar
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Preprocess the frame (convert to B&W and apply Hough transform)
                preprocessed_frame = preprocess_frame(frame)
                
                # Process the frame for segmentation
                segmentation_map = process_frame(preprocessed_frame, model, device)
                
                # Create colored overlay
                overlay = create_overlay(segmentation_map)
                
                # Resize overlay to match original frame size
                overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Blend preprocessed frame with overlay
                result = cv2.addWeighted(preprocessed_frame, 0.7, overlay, 0.3, 0)
                
                # Write the frame to output video
                out.write(result)
                
                pbar.update(1)
            
            except Exception as e:
                logging.error(f"Error processing frame: {e}")
                break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    logging.info(f"Video processing completed. Output saved to: {output_video_path}")

if __name__ == '__main__':
    main()