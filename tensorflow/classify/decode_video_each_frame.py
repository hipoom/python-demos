import subprocess

import ffmpeg


def decode_2_pngs(video_path: str, images_dir: str):
    subprocess.call("ffmpeg -i " + video_path + " img_%d.png", shell=True, cwd=images_dir)


if __name__ == '__main__':
    decode_2_pngs("C:/Users/28496/Desktop/ScreenRecording_1.MP4", "C:/Users/28496/Desktop/temp")