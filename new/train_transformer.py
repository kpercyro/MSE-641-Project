import torch


def train_transformer_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
):
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # ------------------
        # Training
        # ------------------
        model.train()
        running_train_loss = 0.0

        for input_ids, attention_mask, y_batch in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
            )
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # ------------------
        # Validation
        # ------------------
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for input_ids, attention_mask, y_batch in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                y_batch = y_batch.to(device)

                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                )
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

    return train_losses, val_losses
