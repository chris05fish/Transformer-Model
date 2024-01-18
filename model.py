import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector

df = pd.read_csv("nba_games.csv", index_col=0)
#sort columns by date
df = df.sort_values("date")
#reset index to make 0,1,2 at top 
df = df.reset_index(drop=True)
#delete unnessesary/identicle rows
del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]
#create a target column that is whether the team won or lost the next game they play
#function that shifts the won/lost column and pull it back one row
def add_target(group):
    group["target"] = group["won"].shift(-1)
    return group

df = df.groupby("team", group_keys=False).apply(add_target)
#make null values in target column == 2
df["target"][pd.isnull(df["target"])] = 2
#convert win/loss from true/false to 1/0
df["target"] = df["target"].astype(int, errors="ignore")
#remove any columns with null values from dataset
nulls = pd.isnull(df).sum()
nulls = nulls[nulls > 0]
valid_columns = df.columns[~df.columns.isin(nulls.index)]
#create new dataframe with copy
df = df[valid_columns].copy()

rr = RidgeClassifier(alpha=1)
#split data
split = TimeSeriesSplit(n_splits=3)
#feature selector to fit the model and prevent overfitting by making smaller data
sfs = SequentialFeatureSelector(rr, 
                                n_features_to_select=30, 
                                direction="forward",
                                cv=split,
                                n_jobs=1
                               )
#feature selection = train model to pick best features as it learns
#dont scale the removed columns because not numbers
#scale columns between 0 and 1 for ridge regression
removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]
scaler = MinMaxScaler()
#scaler.fit(df[selected_columns])
df[selected_columns] = scaler.fit_transform(df[selected_columns])

sfs.fit(df[selected_columns], df["target"])
#get list of columns from feature selector 
predictors = list(selected_columns[sfs.get_support()])
#scaler.fit(df[predictors])

# Define your SportsOutcomePredictor model
class SportsOutcomePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SportsOutcomePredictor, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(0)  # Add a batch dimension
        transformer_output = self.transformer(x, x)
        output = self.output_layer(transformer_output.squeeze(0))  # Remove the batch dimension
        return output

# Create a DataLoader for PyTorch
class SportsDataset(Dataset):
    def __init__(self, data, predictors, scaler):
        self.data = data[predictors] # Use only selected columns
        self.scaler = scaler  # Pass the scaler as an attribute

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract relevant features columns 2nd to last (excluding the 'target' column)
        features = self.data.iloc[idx, 2:-1].values.astype(float)
        #features = torch.tensor(features).float()  #Change .long() to .float()
        #features = self.data.iloc[idx, 2:-1].values.astype(float)

        # Scale the features using the provided MinMaxScaler
        # Create a new scaler for each batch
        batch_scaler = MinMaxScaler()
        features = batch_scaler.fit_transform(features.reshape(1, -1))
        #features = self.scaler.transform(features.reshape(1, -1))

        # Convert the features tensor to the appropriate data type (float32 or int64)
        features = torch.tensor(features.flatten(), dtype=torch.float32)  # or torch.int64

        # Extract the label (target)
        label = int(self.data.iloc[idx, -1])

        return features, torch.tensor(label, dtype=torch.long)

batch_size = 64  # Desired batch size
# Instantiate the dataset and dataloader
dataset = SportsDataset(df, predictors, scaler)  
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Instantiate the model
input_size = len(dataset[0][0])  # Actual size of vocabulary or input features
hidden_size = 64
output_size = 2  # Number of classes in your outcome prediction task
model = SportsOutcomePredictor(input_size, hidden_size, output_size)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        #outputs = model(inputs)
        outputs = model(inputs.view(-1, input_size))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
