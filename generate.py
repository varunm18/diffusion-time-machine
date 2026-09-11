import argparse

import torch
from diffusers import StableDiffusionPipeline


MODEL_ID = "runwayml/stable-diffusion-v1-5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default=(
            "A historical photograph of a famous landmark before its destruction, "
            "documentary style, realistic architecture, natural daylight"
        ),
    )
    parser.add_argument("--output", default="generated.png")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID)
    pipe = pipe.to(device)

    image = pipe(args.prompt).images[0]
    image.save(args.output)
    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()