
# load the replay file and render to video
from env.replay import Replay
import argparse
import imageio
import numpy.typing as npt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # type: ignore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_path', type=str, default='file.replay', help='Replay file to render to video')
    parser.add_argument('--video_path', type=str, default='output.mp4', help='Output file')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    args = parser.parse_args()

    print(f"Loading replay from {args.replay_path}")
    replay = Replay.load(args.replay_path)

    clip = ImageSequenceClip(replay.data["global_obs"], fps=args.fps)
    print(f"Writing video to {args.video_path}")
    clip.write_videofile(args.video_path)
