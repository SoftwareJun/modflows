

class RealDataset(nn.Module):
    def __init__(self, is_train, args):
        
        root = args.data_path if is_train else args.eval_data_path

        self.data_list = []
        
        for filename in os.listdir(root):

            file_path = os.path.join(root, filename)

            if '0_real' not in os.listdir(file_path):
                for folder_name in os.listdir(file_path):
                
                    assert set(os.listdir(os.path.join(file_path, folder_name))) == {'0_real'}

                    for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                        self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
            
            else:
                for image_path in os.listdir(os.path.join(file_path, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : })


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        sample = self.data_list[index]
                
        image_path = sample['image_path']

        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        return mage, torch.tensor(int(targets))i

def save_images(path ):

    for filename in os.listdir(root):

    file_path = os.path.join(root, filename)

    if '0_real' not in os.listdir(file_path):
    for folder_name in os.listdir(file_path):

        assert set(os.listdir(os.path.join(file_path, folder_name))) == {'0_real'}

        os.makedir(os.path.join(file_path, folder_name, "1_fake"), exits_ok=True)
        for img in len(data):
            save(os.path.join(file_path, folder_name, "1_fake"), img_path)
            

    else:
        os.makedir(os.path.join(file_path, folder_name, "1_fake"), exits_ok=True)
        for img in len(data):
            save(os.path.join(file_path, folder_name, "1_fake"), img_path)

def reconstruct_simple(x, ae, seed, steps=None, tools=None):
    decode_dtype = ae.dtype
    generator = torch.Generator().manual_seed(seed)
    x = x.to(dtype=ae.dtype) * 2.0 - 1.0
    latents = retrieve_latents(ae.encode(x), generator=generator)

    reconstructions = ae.decode(
                        latents.to(decode_dtype), return_dict=False
                    )[0]
    reconstructions = (reconstructions / 2 + 0.5).clamp(0, 1)
    return reconstructions


def recon(pipe, dataloader, ae, seed, args, tools):
    for batch_idx, (images, names) in enumerate(dataloader):
        if batch_idx == 0:
            print(f"Batch {batch_idx + 1}:")
            print(f" - Images shape: {images.shape}")
        with torch.no_grad():
            recons = reconstruct_simple(x=images.to(device), ae=ae, seed=seed, steps=args.steps, tools=tools)
        save_images(images_tensor=recons, input_path=args.input_folder, names=names)

def get_vae(repo_id, return_full=False):
    if 'ldm' in repo_id:
        pipe = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256", cache_dir="weights")
        return pipe.vqvae
    elif 'stable-diffusion-3' in repo_id:
        pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16,cache_dir='weights')
        return pipe.vae
        #pipe = pipe.to("cuda")

def main():
    parser ;
    parser.add_argument('--repo_id', type=str, default='stable-diffusion-3'
                        help='Correct stable diffusion autoencoder')
    dataloader = create_dataloader(args.input_folder, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    ae = get_vae(repo_id=args.repo_id).to(device)
    tools = None
    recon(pipe=None, dataloader=dataloader, ae=ae, seed=seed, args=args, tools=tools)

if __name__=='__main__':
    main()
            