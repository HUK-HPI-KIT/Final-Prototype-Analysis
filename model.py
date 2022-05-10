from curses.ascii import CR
import logging
from torch.nn import Module, Sequential, Linear, ReLU, Embedding, LayerNorm, Sigmoid
from torch.nn import BCEWithLogitsLoss
import torch
from transformers import PreTrainedModel, RobertaModel, RobertaConfig

class InsurwayRecommender(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        logging.info("creating model...")

        self.roberta = RobertaModel(config)
        self.fc = Sequential(
            Linear(config.hidden_size, config.hidden_size//2),
            ReLU(),
            Linear(config.hidden_size//2, 1)
        )
        self.loss = BCEWithLogitsLoss()
        self.init_weights()

    def forward(self, **model_input):
        output = self.roberta(model_input["input_ids"], attention_mask=model_input["attention_mask"])
        score = self.fc(output[1])
        
        loss = self.loss(score, model_input["labels"].unsqueeze(1).to(torch.float16))
        # print("loss:", loss.item())
        return (loss, score)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (Linear, Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, Linear) and module.bias is not None:
            module.bias.data.zero_()