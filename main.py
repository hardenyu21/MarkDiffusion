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
    import torch
    torch.manual_seed(42)
    import warnings
    warnings.filterwarnings('ignore')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w_pipe = WaterMarkedStableDiffusionPipeline("CompVis/stable-diffusion-v1-4")
    w_pipe.to(device)
    images, msgs = w_pipe.generate(['A man is walking on the road', 'A man is playing basketball'], return_msg = True)
    for i, img in enumerate(images):
        img.save(f"test{i+2}.png")
        print(msgs[i])