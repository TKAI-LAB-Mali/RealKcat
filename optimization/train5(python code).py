import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel

# Load and preprocess data
df = pd.read_csv('train_pafa_1039.csv', header=None, names=['sequence', 'kcat'])
df['kcat'] = df['kcat'].astype(float)

# Cluster kcat values
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['kcat']])

# Load ESM-2 model and tokenizer
model_name = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
esm_model = AutoModel.from_pretrained(model_name)

# Generate ESM-2 embeddings
def get_esm_embedding(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = esm_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

df['embedding'] = df['sequence'].apply(get_esm_embedding)

# Prepare data for training
X = np.stack(df['embedding'].values)
y = df['cluster'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the fine-tuning model
class FineTuningModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FineTuningModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.selu = nn.SELU()
        
    def telu(self, x):
        return x * torch.tanh(torch.exp(x))

    def telu_grad(self, x):
        return torch.tanh(torch.exp(x)) + x * (1 - torch.tanh(torch.exp(x))**2) * torch.exp(x)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
input_dim = X_train.shape[1]
output_dim = len(np.unique(y))
model = FineTuningModel(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 50000
batch_size = 8

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for i in range(0, len(X_train), batch_size):
        batch_X = torch.FloatTensor(X_train[i:i+batch_size])
        batch_y = torch.LongTensor(y_train[i:i+batch_size])

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == batch_y).sum().item()
        total_predictions += batch_y.size(0)

    epoch_loss = total_loss / (len(X_train) // batch_size)
    epoch_accuracy = correct_predictions / total_predictions

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(torch.FloatTensor(X_test))
    _, predicted = torch.max(test_outputs, 1)
    accuracy = accuracy_score(y_test, predicted)
    print(f'Test Accuracy: {accuracy:.4f}')

# Function to predict kcat range for a new sequence
def predict_kcat_range(sequence):
    embedding = get_esm_embedding(sequence)
    with torch.no_grad():
        output = model(torch.FloatTensor(embedding).unsqueeze(0))
        predicted_cluster = torch.argmax(output).item()
    
    cluster_ranges = df.groupby('cluster')['kcat'].agg(['min', 'max'])
    predicted_range = cluster_ranges.loc[predicted_cluster]
    return predicted_range['min'], predicted_range['max']

# Example usage
new_sequence = "MDIGIDSDPQKTNAVPRPKLVVGLVVDQMRWDYLYRYYSKYGEGGFKRMLNTGYSLNNVHIDYVPTVTAIGHTSIFTGSVPSIHGIAGNDWYDKELGKSVYCTSDETVQPVGTTSNSVGQHSPRNLWSTTVTDQLGLATNFTSKVVGVSLKDRASILPAGHNPTGAFWFDDTTGKFITSTYYTKELPKWVNDFNNKNVPAQLVANGWNTLLPINQYTESSEDNVEWEGLLGSKKTPTFPYTDLAKDYEAKKGLIRTTPFGNTLTLQMADAAIDGNQMGVDDITDFLTVNLASTDYVGHNFGPNSIEVEDTYLRLDRDLADFFNNLDKKVGKGNYLVFLSADHGAAHSVGFMQAHKMPTGFFVEDMKKEMNAKLKQKFGADNIIAAAMNYQVYFDRKVLADSKLELDDVRDYVMTELKKEPSVLYVLSTDEIWESSIPEPIKSRVINGYNWKRSGDIQIISKDGYLSAYSKKGTTHSVWNSYDSHIPLLFMGWGIKQGESNQPYHMTDIAPTVSSLLKIQFPSGAVGKPITEVIGRIEGRSAWSHPQFEK"
min_kcat, max_kcat = predict_kcat_range(new_sequence)
print(f"Predicted kcat range: {min_kcat:.2f} - {max_kcat:.2f}")
