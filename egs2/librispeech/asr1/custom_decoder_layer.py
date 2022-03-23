# Decoder with attentions

import torch
from torch import nn
from espnet2.asr.decoder.transformer_decoder import BaseTransformerDecoder
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

from typeguard import check_argument_types

# similar to DecoderLayer class in espnet.nets.pytorch_backend.transformer.decoder_layer
class MyDecoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        self_attn: nn.Module,
        audio_attn: nn.Module,
        lm_attn: nn.Module,
        feed_forward: nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
        concat_after: bool = False
    ):
        super(MyDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.audio_attn = audio_attn
        self.lm_attn = lm_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.norm4 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)
            self.concat_linear3 = nn.Linear(size + size, size)
    
    def attn_block_foward(attn, norm, concat_linear, x, q, k, v, mask):
        residual = x
        if self.normalize_before:
            x = norm(x)
        if self.concat_after:
            x_concat = torch.cat(
                (x, attn(q, k, v, mask)), dim=-1
            )
            x = residual + concat_linear(x_concat)
        else:
            x = residual + self.dropout(attn(q, k, v, mask))
        if not self.normalize_before:
            x = norm(x)
        return x
         
    def forward(self, tgt, tgt_mask, audio, audio_mask, lmembed, lmembed_mask, cache=None):
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            # query is same as target
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]
        
        if self.concat_after:
            tgt_concat = torch.cat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
            )
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        # attn block for audio
        x = attn_block_foward(
            attn=self.audio_attn,
            norm=self.norm2,
            concat_linear=self.concat_linear2,
            x=x, q=x, k=audio, v=audio,
            mask=audio_mask
        )

        # attn block for lm embeddings
        x = attn_block_foward(
            attn=self.lm_attn,
            norm=self.norm3,
            concat_linear=self.concat_linear3,
            x=x, q=x, k=lmembed, v=lmembed,
            mask=lmembed_mask
        )
        
        residual = x
        if self.normalize_before:
            x = self.norm4(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm4(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        #return x, tgt_mask, memory, memory_mask
        return x, tgt_mask, audio, audio_mask, lmembed, lmembed_mask

# espnet2.asr.decoder.transformer_decoder
class MyDecoder(BaseTransformerDecoder):
    def __init__(
        self,
        vocab_size: int = 0,
        encoder_output_size: int = 0,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        audio_attention_dropout_rate: float = 0.0,
        lm_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        assert check_argument_types()
        # TODO how to init
        super().__init__(
            vocab_size=vocab_size,
            encoder_output_size=encoder_output_size,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            input_layer=input_layer,
            use_output_layer=use_output_layer,
            pos_enc_class=pos_enc_class,
            normalize_before=normalize_before,
        )
        attention_dim = encoder_output_size
        self.decoders = repeat(
            num_blocks,
            lambda lnum: MyDecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, audio_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, lm_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            )
        )

    # replace origin forward and forward_one_step to include lm embed
    # don't change score and batch_score
    def forward(
        self,
        hs_audio_pad,
        h_audio_lens,
        hs_lm_pad,
        h_lm_lens,
        ys_in_pad,
        ys_in_lens
    ):
        # need to create masks for audio, lm, and decoder input
        tgt = ys_in_pad
        tgt_mask = (~make_pad_mask(ys_in_lens)[:, None, :]).to(tgt.device)
        m = subsequent_mask(tgt_mask.size(-1), device=tgt_mask.device).unsqueeze(0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        audio = hs_audio_pad
        audio_mask = (~make_pad_mask(h_audio_lens, maxlen=audio_mask.size(1)))[:, None, :].to(
            audio.device
        )
 
        lmembed = hs_lm_pad
        lm_mask = (~make_pad_mask(h_lm_lens, maxlen=lm_mask.size(1)))[:, None, :].to(
            lmembed.device
        )
        
        x = self.embed(tgt)
        x, tgt_mask, audio, audio_mask, lmembed, lmembed_mask = self.decoders(
            x, tgt_mask, audio, audio_mask, lmembed, lmembed_mask
        )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        return x, olens
    # memory to audio and lmembeds
    def forward_one_step(
        self,
        tgt,
        tgt_mask,
        audio,
        lmembeds,
        cache
    ):
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, audio, audio_mask, lmembed, lmembed_mask = decoder(
                x, tgt_mask, audio, None, lmembed, None, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache
 
