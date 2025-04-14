import torch 
import torch.nn as nn
import torch.nn.functional as F
 
class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 128]):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for (in_size, out_size) in zip(in_sizes, sizes)
        ])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=self.training)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = F.relu(self.linear(x))
        outputs, _ = self.lstm(x)
        return outputs
    
class LocationSensitiveAttention(nn.Module):
    def __init__(self, attention_dim, encoder_dim, decoder_dim):
        super().__init__()
        self.query_layer = nn.Linear(decoder_dim, attention_dim)
        self.memory_layer = nn.Linear(encoder_dim, attention_dim)
        self.location_conv = nn.Conv1d(1, attention_dim, kernel_size=31, padding=15)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, query, memory, attention_weights_cat):
        processed_query = self.query_layer(query.unsqueeze(1))  # (B, 1, A)
        processed_memory = self.memory_layer(memory)  # (B, T, A)
        processed_location = self.location_conv(attention_weights_cat.unsqueeze(1)).transpose(1, 2)  # (B, T, A)

        energies = self.v(torch.tanh(
            processed_query + processed_memory + processed_location
        )).squeeze(-1)  # (B, T)

        attention_weights = F.softmax(energies, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)
        return context, attention_weights


class Decoder(nn.Module):
    def __init__(self, mel_dim, encoder_dim, decoder_dim):
        super().__init__()
        self.prenet = Prenet(mel_dim, [256, 128])
        self.attention_rnn = nn.LSTMCell(128 + encoder_dim, decoder_dim)
        self.attention_layer = LocationSensitiveAttention(128, encoder_dim, decoder_dim)
        self.decoder_rnn = nn.LSTMCell(decoder_dim + encoder_dim, decoder_dim)
        self.mel_proj = nn.Linear(decoder_dim + encoder_dim, mel_dim)
        self.stop_proj = nn.Linear(decoder_dim + encoder_dim, 1)

    def forward(self, encoder_outputs, mel_inputs):
        B, T, _ = mel_inputs.size()
        mel_outputs = []
        stop_tokens = []
        attention_weights_cat = torch.zeros(B, encoder_outputs.size(1)).to(mel_inputs.device)

        h_att, c_att = torch.zeros(B, 1024).to(mel_inputs.device), torch.zeros(B, 1024).to(mel_inputs.device)
        h_dec, c_dec = torch.zeros(B, 1024).to(mel_inputs.device), torch.zeros(B, 1024).to(mel_inputs.device)
        context = torch.zeros(B, encoder_outputs.size(2)).to(mel_inputs.device)

        for t in range(T):
            prenet_out = self.prenet(mel_inputs[:, t])
            att_input = torch.cat((prenet_out, context), dim=-1)
            h_att, c_att = self.attention_rnn(att_input, (h_att, c_att))
            context, attn_weights = self.attention_layer(h_att, encoder_outputs, attention_weights_cat)

            dec_input = torch.cat((h_att, context), dim=-1)
            h_dec, c_dec = self.decoder_rnn(dec_input, (h_dec, c_dec))

            proj_input = torch.cat((h_dec, context), dim=-1)
            mel_frame = self.mel_proj(proj_input)
            stop_token = self.stop_proj(proj_input)

            mel_outputs.append(mel_frame.unsqueeze(1))
            stop_tokens.append(stop_token.squeeze(1))

            attention_weights_cat = attn_weights.detach()

        mel_outputs = torch.cat(mel_outputs, dim=1)
        stop_tokens = torch.stack(stop_tokens, dim=1)
        return mel_outputs, stop_tokens

class Postnet(nn.Module):
    def __init__(self, mel_dim):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(mel_dim, 512, kernel_size=5, padding=2),
                nn.BatchNorm1d(512),
                nn.Tanh(),
                nn.Dropout(0.5))
        )

        for _ in range(3):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(512, 512, kernel_size=5, padding=2),
                    nn.BatchNorm1d(512),
                    nn.Tanh(),
                    nn.Dropout(0.5))
            )

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(512, mel_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(mel_dim),
                nn.Dropout(0.5))
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, mel_dim, T)
        for conv in self.convolutions:
            x = conv(x)
        return x.transpose(1, 2)  # (B, T, mel_dim)


class Tacotron2(nn.Module):
    def __init__(self, input_dim, mel_dim):
        super(Tacotron2, self).__init__()
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=512)
        self.decoder = Decoder(mel_dim=mel_dim, encoder_dim=512, decoder_dim=1024)
        self.postnet = Postnet(mel_dim=mel_dim)

    def forward(self, text_embeddings, mel_inputs):
        encoder_outputs = self.encoder(text_embeddings)
        mel_outputs, stop_tokens = self.decoder(encoder_outputs, mel_inputs)
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)
        return mel_outputs, mel_outputs_postnet, stop_tokens

