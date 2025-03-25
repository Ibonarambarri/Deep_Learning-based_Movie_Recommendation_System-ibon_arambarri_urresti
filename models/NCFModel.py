class ImprovedNCFModel(nn.Module):
    def __init__(self, n_users, n_movies, embedding_dim, hidden_layers, dropout_rate):
        super(ImprovedNCFModel, self).__init__()

        # Capas de embedding con inicialización mejorada
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)

        # Inicialización de Xavier/Glorot para mejorar convergencia
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)

        # Capas del modelo MLP (MultiLayer Perceptron)
        self.mlp_layers = nn.ModuleList()
        input_size = embedding_dim * 2

        for next_size in hidden_layers:
            self.mlp_layers.append(nn.Linear(input_size, next_size))
            self.mlp_layers.append(nn.BatchNorm1d(next_size))  # Añadido BatchNorm
            self.mlp_layers.append(nn.LeakyReLU(0.1))  # LeakyReLU en lugar de ReLU
            self.mlp_layers.append(nn.Dropout(p=dropout_rate))
            input_size = next_size

        # Capa de salida
        self.output_layer = nn.Linear(hidden_layers[-1], 1)

    def forward(self, user_indices, movie_indices):
        # Obtener embeddings
        user_embedding = self.user_embedding(user_indices)
        movie_embedding = self.movie_embedding(movie_indices)

        # Concatenar embeddings
        x = torch.cat([user_embedding, movie_embedding], dim=1)

        # Pasar por las capas MLP
        for layer in self.mlp_layers:
            x = layer(x)

        # Capa de salida
        output = self.output_layer(x)
        return output.squeeze()
