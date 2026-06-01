import os
from fastvideo import VideoGenerator

def main():
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"

    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
    )

    # Define a prompt for your video
    prompt = "a stylish man walks down a istanbul street in Taksim filled with warm glowing neon and animated city signage. He wears a black leather jacket, and black boots. He wears sunglasses. He walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. then pass in front of Galataport, then go to the sultan ahmet and Topkapı palace, watch the see and sunset."

    # Generate the video
    video = generator.generate_video(
        prompt,
        output_path="my_videos/",  # Controls where videos are saved
        save_video=True
    )

if __name__ == '__main__':
    main()
