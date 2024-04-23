if __name__ == "__main__":

    import torch.utils
    import torch.utils.data
    import CNN
    import torch
    import pandas as pd
    import torch.nn as nn
    from batch_dataset import CustomDataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import os

    # Define the training mode
    torch.backends.cudnn.benchmark = True
    EXISTING_MODEL = True
    MODEL_PATH = "model.pth"
    HDF5_PATH = "./melanoma.h5"
    CONCATENATE_EPOCH_LOSSES = True

    # Define the training parameters
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.01
    BATCH_SIZE = 2000

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Cria uma instância do conjunto de dados personalizado
    custom_dataset = CustomDataset(HDF5_PATH)

    # Usa o conjunto de dados personalizado no DataLoader
    train_loader = DataLoader(
        custom_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
    )

    if EXISTING_MODEL:
        # Load the existing model
        model = CNN.CNN()
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        # Create an instance of the CNN model
        model = CNN.CNN()

    # Move the model to the device
    model.to(device)

    # Define the loss function
    criterion = nn.BCELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model

    print("Training the model...\n")
    epoch_losses = []

    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    pbEpochs = tqdm(
        range(NUM_EPOCHS),
        desc=f"{GREEN}Epochs{RESET}",
        colour="green",
    )

    for epoch in pbEpochs:
        pbBatches = tqdm(
            train_loader, desc=f"{RED}Batches{RESET}", leave=False, colour="red"
        )
        for data, label in pbBatches:
            # Move batch data to device (e.g. GPU)
            batch_data = data.to(device)
            batch_labels = label.to(device)

            # Forward pass
            outputs = model(batch_data)

            # Compute the loss
            loss = criterion(outputs, batch_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbBatches.set_postfix({"Batch loss": loss.item()})

        epoch_losses.append(loss.item())
        pbEpochs.set_postfix({"Epoch loss": loss.item()})
    # Close the dataset
    custom_dataset.close()

    # Print the final loss
    print(f"\n\nFinal loss: {loss.item()}\n")

    # Save the losses to a CSV file

    if not CONCATENATE_EPOCH_LOSSES:
        try:
            os.remove("losses.csv")
        except FileNotFoundError:
            pass
    else:
        try:
            loss_df = pd.read_csv("losses.csv")
            epoch_losses = list(loss_df["loss"]) + epoch_losses
        except FileNotFoundError:
            loss_df = pd.DataFrame(epoch_losses, columns=["loss"])
            loss_df.to_csv("losses.csv", index=False)

    # Save the model
    torch.save(model.state_dict(), "model.pth")
