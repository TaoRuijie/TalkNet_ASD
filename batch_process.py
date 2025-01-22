import os
import subprocess
import argparse

def main(args):
    # Ensure the output directory exists
    if not os.path.exists(args.videoFolderOutput):
        os.makedirs(args.videoFolderOutput)
    
    # List all videos in the input directory
    video_files = [f for f in os.listdir(args.videoFolderInput) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"No video files found in the directory: {args.videoFolderInput}")
        return

    # Process each video
    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]

        output_video_path = args.videoFolderOutput

        # Ensure output directory for the video exists
        if not os.path.exists(output_video_path):
            os.makedirs(output_video_path)

        # Build the command to call demoTalkNet.py
        command = [
            "python", "demoTalkNet.py",
            "--videoName", video_name,
            "--videoFolderInput", args.videoFolderInput,
            "--videoFolderOutput", args.videoFolderOutput,
            "--channelName", args.channelName,
        ]

        # Print and execute the command
        print(f"Processing video: {video_file}")
        print("Command:", " ".join(command))
        subprocess.run(command)

    print("Batch processing completed.")

if __name__ == "__main__":
    # Parse arguments for the batch process
    parser = argparse.ArgumentParser(description="Batch Process Videos with demoTalkNet")
    parser.add_argument('--videoFolderInput', type=str, required=True, help='Path to the folder containing input videos.')
    parser.add_argument('--videoFolderOutput', type=str, help='Path to the folder for storing outputs and temporary files.')
    parser.add_argument('--bucketName', type=str, help='Path to the folder for storing outputs and temporary files.')
    parser.add_argument('--channelName', type=str, required=True, help='Path to the folder for storing outputs and temporary files.')
    parser.add_argument('--pretrainModel', type=str,default="pretrain_TalkSet.model", help='Path to the pretrained TalkNet model.')
    parser.add_argument('--fps', type=float, default=25, help='Desired FPS.')
    parser.add_argument('--frame_size', type=int, default=512, help='Desired frame size.')
    parser.add_argument('--angleThreshold', type=int, default=10, help='Yaw threshold.')
    parser.add_argument('--contentDetectorThreshold', type=float, default=27.0, help='Content detector threshold.')
    parser.add_argument('--thresholdDetectorThreshold', type=float, default=30.0, help='Threshold detector threshold.')
    parser.add_argument('--nDataLoaderThread', type=int, default=10, help='Number of data loader threads.')
    parser.add_argument('--facedetScale', type=float, default=0.25, help='Face detection scale factor.')
    parser.add_argument('--minTrack', type=int, default=40, help='Minimum frames for each shot.')
    parser.add_argument('--numFailedDet', type=int, default=5, help='Missed detections allowed before stopping tracking.')
    parser.add_argument('--minFaceSize', type=int, default=100, help='Minimum face size in pixels.')
    parser.add_argument('--cropScale', type=float, default=0.40, help='Scale bounding box.')
    parser.add_argument('--start', type=int, default=0, help='Start time of the video.')
    parser.add_argument('--duration', type=int, default=0, help='Duration of the video (0 for full video).')
    parser.add_argument('--evalCol', action='store_true', help='Evaluate on Columbia dataset.')
    parser.add_argument('--colSavePath', type=str, default="/data08/col", help='Path for inputs, temps, and outputs for Columbia evaluation.')

    args = parser.parse_args()
    main(args)
