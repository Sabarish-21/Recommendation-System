import numpy as np
from tensorflow.keras.layers import Input, Embedding, Dot, Flatten
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Sample data (user-item interactions)
num_users = 1000
num_items = 500
ratings = np.random.randint(1, 6, size=(num_users, num_items))  # Random ratings from 1 to 5
user_ids = np.arange(num_users)
item_ids = np.arange(num_items)

# Split data into training and test sets
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2, random_state=42)

# Define model architecture for matrix factorization
num_factors = 50  # Latent factors dimension
user_input = Input(shape=(1,))
item_input = Input(shape=(1,))
user_embedding = Embedding(num_users, num_factors, embeddings_initializer='he_normal')(user_input)
item_embedding = Embedding(num_items, num_factors, embeddings_initializer='he_normal')(item_input)
user_flat = Flatten()(user_embedding)
item_flat = Flatten()(item_embedding)
prediction = Dot(axes=1)([user_flat, item_flat])  # Dot product of user and item embeddings
model = Model(inputs=[user_input, item_input], outputs=prediction)

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit([user_ids[train_ratings[:, 0]], item_ids[train_ratings[:, 1]]],
          train_ratings[:, 2],
          batch_size=64,
          epochs=10,
          validation_split=0.1)

# Evaluate the model on test data
test_loss = model.evaluate([user_ids[test_ratings[:, 0]], item_ids[test_ratings[:, 1]]],
                            test_ratings[:, 2])

print(f'Test Loss: {test_loss}')

# Make predictions for a specific user
user_id = 10
item_ids_to_predict = np.arange(num_items)
user_ids_to_predict = np.full_like(item_ids_to_predict, user_id)
predicted_ratings = model.predict([user_ids_to_predict, item_ids_to_predict])

# Get top N recommendations
N = 10
top_indices = np.argsort(predicted_ratings[:, 0])[::-1][:N]
top_item_ids = item_ids_to_predict[top_indices]
top_ratings = predicted_ratings[top_indices, 0]

print(f'Top {N} Recommendations for User {user_id}:')
for item_id, rating in zip(top_item_ids, top_ratings):
    print(f'Item ID: {item_id} | Predicted Rating: {rating}')
