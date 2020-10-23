from torch import nn, optim
from torch.utils.data import DataLoader

def train(model, dataset, epochs, bs, seq_len, cuda):
    """Trains a model with a dataset.
    Takes the number of epochs, batch size, and sequence length."""

    # Set model to training mode
    model.train()
    
    # Initialize necessary classes
    dataloader = DataLoader(dataset, batch_size=bs)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Iterate for num epochs
    for epoch in range(epochs):

        # Get the initial LSTM state
        state_h, state_c = model.init_state(seq_len)

        # Vars to calculate cost
        cost, num = 0, 0

        # Iterate data batches
        for batch, (x, y) in enumerate(dataloader):

            # Use GPU if desired
            if cuda:
                x, y = x.to('cuda'), y.to('cuda')
                state_h, state_c = state_h.to('cuda'), state_c.to('cuda')

            # Set the gradients to zero
            optimizer.zero_grad()

            # Run the forward pass
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            # Detach hidden states
            state_h, state_c = state_h.detach(), state_c.detach()

            # Run back-propigation
            loss.backward()

            # Adjust the weights
            optimizer.step()

            # Vars for calculating cost
            cost += loss.item()
            num += 1

            print('epoch:', epoch, ' batch:', batch, ' loss:', loss.item())

        print('epoch:', epoch, ' cost:', cost/num)
