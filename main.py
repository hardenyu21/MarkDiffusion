if __name__ == '__main__':

    """

    pipe = WaterMarkedStableDiffusionPipeline("CompVis/stable-diffusion-v1-4", "SS", fine_tune = True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(pipe.vae.encoder.training) 
    print(pipe.vae.decoder.training)
    print(pipe.unet.training)  
    print(pipe.text_encoder.training) 
    """

    #pipe.to(device)

    #prompts = ["A sunny beach", "A snowy mountain"]

    #images, messages = pipe.generate(prompts)
    """
    for i, img in enumerate(images):
        img.save(f"test{i + 2}.png")
        msg = (messages[i] > 0).tolist()
        msg = msg2str(msg)
        print(msg)
    """

    from src.pipeline.stablediffusion import WaterMarkedStableDiffusionPipeline
    from diffusers import StableDiffusionPipeline
    import torch
    torch.manual_seed(42)
    import warnings
    warnings.filterwarnings('ignore')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w_pipe = WaterMarkedStableDiffusionPipeline("CompVis/stable-diffusion-v1-4")
    w_pipe.to(device)
    w_images, msgs = w_pipe.generate(['A man is walking on the road'], return_msg = True)
    non_w_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    non_w_pipe.to(device)
    non_w_images = non_w_pipe(['A man is walking on the road'])[0]
    
    for w_image in w_images:
        w_image.save("w_image.png")
    for non_w_image in non_w_images: 
        non_w_image.save("non_w_image.png")