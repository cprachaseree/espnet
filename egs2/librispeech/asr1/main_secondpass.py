from mydataloader import get_my_dataloader, my_collate_fn
from custom_decoder_layer import MyDecoder
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


def train(decoder, dataloader, epochs):
    for epoch in range(1, epochs+1):
        for data in dataloader:
            print(data)
            out = decoder(data)
            break
        break

def main():
    train_data_path = "/scratch/prac0003/secondpass_exp/data/"
    validation_paths = []
    train_batch_size = 16
    train_epochs = 4
    train_data_loader = get_my_dataloader(train_data_path, train_batch_size, shuffle=True, collate_fn=my_collate_fn)
    decoder = MyDecoder(
        vocab_size = 5000,
        encoder_output_size = 512,
        attention_heads = 4,
        linear_units = 2048,
        num_blocks = 6,
        dropout_rate = 0.1,
        positional_dropout_rate = 0.1,
        self_attention_dropout_rate = 0.0,
        audio_attention_dropout_rate = 0.0,
        lm_attention_dropout_rate = 0.0,
        input_layer = "embed",
        use_output_layer =  True,
        pos_enc_class=PositionalEncoding,
        normalize_before = True,
        concat_after = False
    )
    train(decoder, train_data_loader, train_epochs)


if __name__=="__main__":
    main()
