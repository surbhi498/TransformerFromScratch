import torch
import torch.nn as nn
import math

class InputEmbeddings(torch.nn.Module):
    def __init__(self, d_model: int, vocab_size : int):
        """
        Initializes the InputEmbeddings module.

        Args:
            d_model (int): The dimension of the model.
            vocab_size (int): The size of the vocabulary.

        Returns:
            None
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass of the InputEmbeddings module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, sequence_length, d_model).
        """
        return self.embed(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):    
    def __init__(self, d_model: int, seq_len: int, dropout: float)->None:
        """
        Initializes the PositionalEncoding module.

        Args:
            d_model (int): The dimension of the model.
            seq_len (int): The length of the sequence.
            dropout (float): The dropout probability.

        Returns:
            None

        This function initializes the PositionalEncoding module. It creates a parameter tensor `pe` of shape (1, seq_len, d_model)
        using the given `d_model` and `seq_len`. It also creates a vector of length (seq_len, 1) and applies sin to even position and
        cos to odd position to create the `pe` tensor. Finally, it adds a batch dimension to the `pe` tensor and registers it as a buffer
        using the `register_buffer` method.
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # create a parameter tensor of shape (1, seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of length (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply sin to even position and cos to odd position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # apply unsqueeze to add batch dimension
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # register the parameter tensor saved in pe
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Perform a forward pass of the PositionalEncoding module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Transformed tensor after adding positional encoding and applying dropout.
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
      def __init__(self, eps: float = 1e-6)->None:
        """
        Initializes the LayerNormalization module.

        Args:
            eps (float, optional): The small value added for numerical stability. Defaults to 10**-6.

        Returns:
            None

        This function initializes the LayerNormalization module. It sets the `eps` attribute to the given value and initializes
        the `alpha` and `bias` parameters as torch.nn.Parameter objects with ones and zeros respectively.
        """
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # to be multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # to be added

      def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the LayerNormalization module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The transformed tensor after normalization.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """
        Initializes the FeedForwardBlock module.

        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feed forward layer.
            dropout (float): The dropout probability.

        Returns:
            None

        This function initializes the FeedForwardBlock module. It creates a Linear layer `linear1` with input size `d_model` and output
        size `d_ff`. It then applies a ReLU activation function to the output of `linear1`. It then creates a Linear layer `linear2` with
        input size `d_ff` and output size `d_model`. Finally, it applies a dropout layer with probability `dropout`.
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:   
        """
        Apply a feed forward transformation to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: Transformed tensor after applying feed forward operation.
        """
        # x: (batch_size, sequence_length, d_model)--> (batch_size, sequence_length, d_ff)--> (batch_size, sequence_length, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) 

class MultiHeadAttentionBlock(nn.Module):
    def ___init__(self, d_model: int, num_heads: int, dropout: float):
        """
        Initializes the MultiHeadAttention module.

        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of heads.
            dropout (float): The dropout probability.

        Returns:
            None

        This function initializes the MultiHeadAttention module. It creates a Linear layer `W_q` with input size `d_model` and output
        size `d_model`. It then creates a Linear layer `W_k` with input size `d_model` and output size `d_model`. It then creates a Linear
        layer `W_v` with input size `d_model` and output size `d_model`. It then creates a Linear layer `W_o` with input size `d_model`
        and output size `d_model`. Finally, it creates a dropout layer with probability `dropout`.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)    
    
    @staticmethod
    def attention(self, q, k, v, mask, dropout: nn.Dropout):
        """
        Compute the attention scores between query and key tensors and return the weighted sum of value tensor.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, num_heads, sequence_length, d_k).
            k (torch.Tensor): Key tensor of shape (batch_size, num_heads, sequence_length, d_k).
            v (torch.Tensor): Value tensor of shape (batch_size, num_heads, sequence_length, d_k).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, sequence_length). Defaults to None.
            dropout (nn.Dropout, optional): Dropout layer to apply to attention scores. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_heads, sequence_length, d_k).
            torch.Tensor: Attention scores tensor of shape (batch_size, num_heads, sequence_length, sequence_length).

        Note:
            The `d_k` attribute of the module is used to determine the dimensionality of the key, query, and value tensors.
        """
        d_k = q.shape[-1]
        # (batch_size, num_heads, sequence_length, d_k)->(batch_size, num_heads, sequence_length, sequence_length)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(attention_scores, dim=-1) # (batch_size, num_heads, sequence_length, sequence_length)
        if dropout is not None:
           attention_scores = dropout(attention_scores)
        output = torch.matmul(attention_scores, v)
        return output, attention_scores
        
          
          
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the forward pass of the MultiHeadAttentionBlock module.

        Args:
            q (torch.Tensor): The query tensor of shape (batch_size, sequence_length, d_model).
            k (torch.Tensor): The key tensor of shape (batch_size, sequence_length, d_model).
            v (torch.Tensor): The value tensor of shape (batch_size, sequence_length, d_model).
            mask (torch.Tensor, optional): The mask tensor of shape (batch_size, sequence_length).
                Defaults to None.

        Returns:
             torch.Tensor: The output tensor of shape (batch_size, sequence_length, d_model).

        This function performs the forward pass of the MultiHeadAttention module. It takes in the query, key, and value tensors,
        as well as a mask tensor. It applies linear transformations to the query, key, and value tensors using the
        `W_q`, `W_k`, and `W_v` linear layers respectively. It then reshapes the tensors to have the shape
        (batch_size, num_heads, sequence_length, d_k) and permutes the dimensions. Finally, it computes the dot product
        attention between the query, key, and value tensors, and applies a mask to the output.

        Note:
            The `d_model` and `num_heads` attributes of the module are used to determine the number of heads and the
            dimensionality of the key, query, and value tensors.
        """
        # Apply linear transformations to the query, key, and value tensors
        query = self.W_q(q) # (batch_size, sequence_length, d_model) --> (batch_size, sequence_length, d_model)
        key = self.W_k(k) # (batch_size, sequence_length, d_model) --> (batch_size, sequence_length, d_model)
        value = self.W_v(v) # (batch_size, sequence_length, d_model) --> (batch_size, sequence_length, d_model)

        # Reshape and transpose the tensors to have the shape (batch_size, num_heads, sequence_length, d_k)
        # (batch_size, sequence_length, d_model) --> (batch_size, sequence_length, num_heads, d_k) --> (batch_size, num_heads, sequence_length, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # Compute the dot product attention between the query, key, and value tensors
        # and apply a mask to the output
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Reshape and transpose the output tensor to have the shape (batch_size, sequence_length, d_model)
        # (batch_size, num_heads, sequence_length, d_k) --> (batch_size, sequence_length, num_heads * d_k) --> (batch_size, sequence_length, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)
        return self.W_o(x)
    
class ResidualConnectionLayer(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    def forward(self, x: torch.Tensor, sublayer: torch.Tensor) -> torch.Tensor:  
        """
        Perform the forward pass of the ResidualConnectionLayer module.

        Args:
            x (torch.Tensor): The input tensor.
            sublayer (torch.Tensor): The sublayer to be applied to x.

        Returns:
            torch.Tensor: The transformed tensor after applying the residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout: float)->None:
        """
        Initializes an instance of the EncoderBlock class.

        Args:
            self_attention_block (MultiHeadAttentionBlock): An instance of the MultiHeadAttentionBlock class.
            feed_forward_block (FeedForwardBlock): An instance of the FeedForwardBlock class.
            dropout (float): The dropout probability.

        Returns:
            None

        This function initializes an instance of the EncoderBlock class. It sets the `self_attention_block` attribute to the provided
        `self_attention_block` instance, the `feed_forward_block` attribute to the provided `feed_forward_block` instance, and the
        `residual_connection_layer` attribute to a list of two instances of the ResidualConnectionLayer class, each initialized with the
        provided `dropout` value.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_layer = nn.ModuleList([ResidualConnectionLayer(dropout) for _ in range(2)]) # ResidualConnectionLayer(dropout) 
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: 
        """
        Perform the forward pass of the EncoderBlock module.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The mask tensor.

        Returns:
            torch.Tensor: The transformed tensor after applying the residual connection.
        """
        x = self.residual_connection_layer[0](x, lambda x: self.self_attention_block(x, x, x, mask))  
        x = self.residual_connection_layer[1](x, lambda x: self.feed_forward_block(x))  
        return x  

class Encoder(nn.Module):
    def __init__(self, layers : nn.ModuleList)->None:
        """
        Initializes an instance of the class.

        Args:
            layers (nn.ModuleList): A list of PyTorch modules.

        Returns:
            None
        """
        super().__init__()
        self.layers = layers    
        self.norm = LayerNormalization()
   
    def forward(self, x, mask):
        """
        Perform a forward pass through the layers.

        Args:
            x (torch.Tensor): The input tensor.
            mask (torch.Tensor): The mask tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the layers.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)       

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, cross_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout: float)->None: 
        """
        Initializes the DecodeBlock module.

        Args:
            self_attention_block (MultiHeadAttentionBlock): The self-attention block.
            cross_attention_block (MultiHeadAttentionBlock): The cross-attention block.
            feed_forward_block (FeedForwardBlock): The feed-forward block.
            dropout (float): The dropout probability.

        Returns:
            None

        This function initializes the DecodeBlock module. It sets the self-attention block, cross-attention block,
        feed-forward block, and creates a ModuleList of residual connection layers with the given dropout for each layer.
        """
        super.__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection_layer = nn.ModuleList([ResidualConnectionLayer(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Perform the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            encoder_output (torch.Tensor): The output of the encoder.
            src_mask (torch.Tensor): The source mask.
            tgt_mask (torch.Tensor): The target mask.

        Returns:
            torch.Tensor: The output tensor.

        This function performs the forward pass of the model. It applies the self-attention block, cross-attention block,
        and feed-forward block to the input tensor. The output of each block is passed through the corresponding
        residual connection layer. The final output is returned.
        """
        x = self.residual_connection_layer[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection_layer[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection_layer[2](x, lambda x: self.feed_forward_block(x))
        return x    

class Decoder(nn.Module):
    def __init__(self, layers : nn.ModuleList)->None:
        """
        Initializes the class with the given layers and a LayerNormalization object.
        
        Args:
            layers (nn.ModuleList): The layers to be initialized.
        
        Returns:
            None
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Perform the forward pass of the model. It applies the self-attention block, cross-attention block,
        and feed-forward block to the input tensor. The output of each block is passed through the corresponding
        residual connection layer. The final output is returned.
        """
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)  

class ProjectionLayer(nn.Module):
       def __init__(self, d_model : int, d_vocab : int)->None:
           """
           Initializes the ProjectionLayer class with the given dimensions.
           Args:
               d_model (int): The dimension of the model.
               d_vocab (int): The dimension of the vocabulary.
           Returns:
               None
           """
           super().__init__()
           self.proj = nn.Linear(d_model, d_vocab)
        
       def forward(self, x)->None:
            """
            Perform the forward pass of the model.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_len, d_vocab) after applying the log softmax function along the last dimension.
            """
            #(batch_size, seq_len, d_model) -> (batch_size, seq_len, d_vocab)
            return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder : Encoder, decoder : Decoder, src_embed : InputEmbeddings, tgt_embed : InputEmbeddings, src_pos : PositionalEncoding, tgt_pos : PositionalEncoding, projection_layer : ProjectionLayer)->None:
        """
        Initializes the Transformer class with the given encoder, decoder, and projection layer.

        Args:
            encoder (Encoder): The encoder module.
            decoder (Decoder): The decoder module.
            projection_layer (ProjectionLayer): The projection layer.

        Returns:
            None
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer    
    
    def encode(self, src, src_mask):
        """
        Encodes the input source sequence using the Transformer encoder.

        Args:
            src (torch.Tensor): The input source sequence of shape (batch_size, sequence_length).
            src_mask (torch.Tensor): The mask tensor for the source sequence of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The encoded source sequence of shape (batch_size, sequence_length, d_model).
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        # (batch, seq_len, d_model)
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, encode_output, src_mask, tgt, tgt_mask):
        """
        Decodes the target sequence using the Transformer decoder.

        Args:
            encode_output (torch.Tensor): The output of the encoder.
            src_mask (torch.Tensor): The mask for the source sequence.
            tgt (torch.Tensor): The target sequence to be decoded.
            tgt_mask (torch.Tensor): The mask for the target sequence.

        Returns:
            torch.Tensor: The decoded output of the Transformer decoder.

        This function takes the encoded output of the encoder, the source mask, the target sequence to be decoded, and the target mask as input.
        It embeds the target sequence using the target embedding layer and applies positional encoding to it.
        Then, it passes the embedded target sequence, the encoded output, the source mask, and the target mask to the Transformer decoder.
        The decoder generates the decoded output based on the input.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        # (batch, seq_len, d_model)
        return self.decoder(tgt, encode_output, src_mask, tgt_mask)   
    
    def project(self, x):
        """
        Projects the input tensor x using the projection_layer.

        Args:
            x: Input tensor to be projected.

        Returns:
            Tensor: The projected tensor.
        """
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)      

def buld_transformer(src_vocab_size : int, tgt_vocab_size : int, src_seq_len : int, tgt_seq_len : int, d_model : int=512, N : int=6, heads : int=8, d_ff : int=2048, dropout : float=0.1)->Transformer:
    """
    Builds a transformer model with the given parameters.

    Args:
        src_vocab_size (int): The size of the source vocabulary.
        tgt_vocab_size (int): The size of the target vocabulary.
        src_seq_len (int): The length of the source sequence.
        tgt_seq_len (int): The length of the target sequence.
        d_model (int, optional): The dimension of the model. Defaults to 512.
        N (int, optional): The number of layers in the encoder and decoder. Defaults to 6.
        heads (int, optional): The number of heads in the multi-head attention block. Defaults to 8.
        d_ff (int, optional): The dimension of the feed-forward layer. Defaults to 2048.
        dropout (float, optional): The dropout probability. Defaults to 0.1.

    Returns:
        Transformer: The transformer model with the given parameters.
    """
    # create the input embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    # create the encoder and decoder blocks
    encoder_blocks = []
    
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block,feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # create the transformer model
    transformer =Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)   
    
    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  
    return transformer        
           
               

      
        
        
          
                  
        
        