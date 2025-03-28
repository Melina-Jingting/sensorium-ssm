def minimal_train_loop(config: dict, save_dir, train_splits: list[str], val_splits: list[str]):
    # Initialize wandb and get device
    wandb.init(project="sensorium_ssm", config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model directly
    ssm_model = DwiseNeuroSSM(**config["nn_module"][1])
    ssm_model.to(device)

    # Weight initialization (if needed)
    if config.get("init_weights", False):
        print("Initializing weights...")
        init_weights(ssm_model)

    # Dataset processing
    indexes_generator = IndexesGenerator(**config["frame_stack"])
    inputs_processor = get_inputs_processor(*config["inputs_processor"])
    responses_processor = get_responses_processor(*config["responses_processor"])
    cutmix = CutMix(**config["cutmix"])

    # Build training dataset
    train_datasets = []
    mouse_epoch_size = config["train_epoch_size"] // constants.num_mice
    for mouse in constants.mice:
        train_datasets.append(
            TrainMouseVideoDataset(
                mouse_data=get_mouse_data(mouse=mouse, splits=train_splits),
                indexes_generator=indexes_generator,
                inputs_processor=inputs_processor,
                responses_processor=responses_processor,
                epoch_size=mouse_epoch_size,
                mixer=cutmix,
            )
        )
    train_dataset = ConcatMiceVideoDataset(train_datasets)
    
    # Build validation dataset
    val_datasets = []
    for mouse in constants.mice:
        val_datasets.append(
            ValMouseVideoDataset(
                mouse_data=get_mouse_data(mouse=mouse, splits=val_splits),
                indexes_generator=indexes_generator,
                inputs_processor=inputs_processor,
                responses_processor=responses_processor,
            )
        )
    val_dataset = ConcatMiceVideoDataset(val_datasets)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_dataloader_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"] // config["iter_size"],
        shuffle=False,
        num_workers=config["num_dataloader_workers"],
    )

    # Optimizer, scheduler, and loss
    optimizer = optim.Adam(ssm_model.parameters(), lr=config["base_lr"])
    total_iterations = len(train_loader) * sum(config["num_epochs"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iterations, eta_min=get_lr(config["min_base_lr"], config["batch_size"])
    )
    loss_fn = MicePoissonLoss()  # Replace with the correct loss function if different
    correlation_metric = CorrelationMetric()
    
    # Training loop
    num_total_epochs = sum(config["num_epochs"])
    global_step = 0
    iter_size = config.get("iter_size", 1)  # Gradient accumulation
    grad_scaler = torch.amp.GradScaler("cuda",enabled=True)  # Mixed precision

    for num_epochs, stage in zip(config["num_epochs"], config["stages"]):
        
        num_iterations = (len(train_dataset) // config["batch_size"]) * num_epochs
        if config.get("warmup", False):
            scheduler = LambdaLR(optimizer, lr_lambda=lambda x: x / num_iterations)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=get_lr(config["min_base_lr"], config["batch_size"]))

        for epoch in range(num_epochs):
            ssm_model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for i, batch in enumerate(train_loader):
                inputs, target = batch
                inputs, target = inputs.to(device), target.to(device)

                with torch.amp.autocast('cuda', enabled=True):                
                    prediction = ssm_model(inputs)
                    loss = loss_fn(prediction, target) / iter_size  # Scale loss for accumulation

                grad_scaler.scale(loss).backward()
                epoch_loss += loss.item() * iter_size

                if (i + 1) % iter_size == 0:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    wandb.log({"train_loss": loss.item() * iter_size, "lr": optimizer.param_groups[0]["lr"], "epoch": epoch + 1, "global_step": global_step})

            # Validation step
            ssm_model.eval()
            val_loss = 0.0
            correlation_metric.reset()
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs, target = batch
                    inputs, target = inputs.to(device), target.to(device)
                    prediction = ssm_model(inputs)
                    loss = loss_fn(prediction, target)
                    val_loss += loss.item()
                    correlation_metric.update({"prediction": prediction, "target": target})
            
            val_loss /= len(val_loader)
            val_corr = correlation_metric.compute()
            avg_corr = np.mean(list(val_corr.values()))  # Get overall mean correlation
            
            print(f"Epoch {epoch+1}/{num_total_epochs} - Train Loss: {epoch_loss/len(train_loader):.4f} - Val Loss: {val_loss:.4f} - Corr: {avg_corr:.4f}")
            wandb.log({
                "epoch_train_loss": epoch_loss / len(train_loader),
                "epoch_val_loss": val_loss,
                "epoch_correlation": avg_corr,
                "epoch": epoch + 1
            })
            # Save model checkpoint
            torch.save(ssm_model.state_dict(), save_dir / f"model_epoch_{epoch+1:03d}.pth")

    wandb.finish()