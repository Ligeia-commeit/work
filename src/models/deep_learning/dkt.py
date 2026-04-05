import numpy as np
import torch
import torch.nn as nn


class DKT(nn.Module):
    def __init__(self, num_items, num_skills, hidden_dim=128, dropout=0.0):
        """
        Deep Knowledge Tracing model.

        Args:
            num_items: Number of items.
            num_skills: Number of skills (kept for API compatibility).
            hidden_dim: Hidden dimension.
            dropout: LSTM dropout.
        """
        super(DKT, self).__init__()
        self.num_items = num_items
        self.num_skills = num_skills
        self.hidden_dim = hidden_dim

        # Interaction encoding: item_id + correctness * num_items
        self.input_embedding = nn.Embedding(2 * num_items, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.output_layer = nn.Linear(hidden_dim, num_items)
        self.sigmoid = nn.Sigmoid()

        # Fallback probability for current simplified predict interface
        self.global_p = 0.5

    def forward(self, input_seq, hidden=None):
        """
        Args:
            input_seq: (batch_size, seq_len)
        Returns:
            output: (batch_size, seq_len, num_items)
            hidden: LSTM hidden state
        """
        embedded = self.input_embedding(input_seq)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.sigmoid(self.output_layer(lstm_out))
        return output, hidden

    def fit(self, data, q_matrix=None, epochs=10, batch_size=32, learning_rate=0.001):
        """
        Train with interaction triplets [student_id, item_id, score].
        """
        data = np.asarray(data)
        if data.size == 0:
            return
        self.global_p = float(np.mean(data[:, 2]))

        # Build student sequences
        sequences = []
        student_ids = np.unique(data[:, 0])
        for student_id in student_ids:
            student_data = data[data[:, 0] == student_id]
            # NOTE: no timestamp in current interface, use item_id order as fallback
            student_data = student_data[student_data[:, 1].argsort()]
            item_ids = student_data[:, 1].astype(int)
            scores = student_data[:, 2].astype(int)
            if len(item_ids) > 1:
                sequences.append({"item_ids": item_ids, "scores": scores})

        if not sequences:
            print("No valid sequences found for training")
            return

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        self.train()

        for epoch in range(epochs):
            total_loss = 0.0
            np.random.shuffle(sequences)

            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i : i + batch_size]
                max_seq_length = max(len(seq["item_ids"]) for seq in batch_sequences)
                if max_seq_length <= 1:
                    continue

                input_seq = torch.zeros((len(batch_sequences), max_seq_length - 1), dtype=torch.long)
                next_item_seq = torch.zeros((len(batch_sequences), max_seq_length - 1), dtype=torch.long)
                target_seq = torch.zeros((len(batch_sequences), max_seq_length - 1), dtype=torch.float32)

                for j, seq in enumerate(batch_sequences):
                    seq_length = len(seq["item_ids"])
                    for k in range(seq_length - 1):
                        item_id = int(seq["item_ids"][k]) % self.num_items
                        score = int(seq["scores"][k])
                        input_seq[j, k] = item_id + score * self.num_items
                    next_item_seq[j, : seq_length - 1] = torch.tensor(
                        seq["item_ids"][1:] % self.num_items, dtype=torch.long
                    )
                    target_seq[j, : seq_length - 1] = torch.tensor(seq["scores"][1:], dtype=torch.float32)

                output, _ = self.forward(input_seq)
                pred_seq = output.gather(2, next_item_seq.unsqueeze(-1)).squeeze(-1)
                loss = criterion(pred_seq.reshape(-1), target_seq.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    def predict(self, student_ids, item_ids, q_matrix=None):
        """
        Current project interface does not provide sequence context at inference.
        Return a calibrated global probability as fallback.
        """
        _ = student_ids, item_ids, q_matrix
        return np.full(len(student_ids), self.global_p, dtype=float)