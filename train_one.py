from src.model.generator 


#encoder is already defined in the main_train.py loop outside here
def train_one_epoch(model, criterion, optimizer, encoder):

    for batch in epoch :
        
        with torch.no_grad:
            # so first image is feature extracted(its now latent image)
            latent = encoder(batch):

            z = gen_preprocess(latent, crop).to(device)
            z = z.unsqueeze(0)
            base_e = generator(z).flatten() #weights for NeuralODE
            base_encoded_flow = NeuralODE(input_dim=3, hidden=generator.hidden, device=device)
            base_encoded_flow.set_weights(base_e)

            # now the latent image is sent to the uniform space by rectified flow
            z = base_encoded_flow(latent)

        pred = model(z)
        
        loss(pred, target)


        
        
        