import random
import torch
import models
import torch.nn as nn
from models.base_model import CaptionModel
from models.utils import repeat_tensor
import models.decoder
from utils.utils_temporal.logsoft_max_utils import linear_softmax_pool_with_lens

class TransformerTemporalModel(CaptionModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                models.decoder.TransformerDecoder,
                models.decoder.TransformerTemporalDecoder,
            )
        super().__init__(encoder, decoder, **kwargs)
    
    def forward(self, input_dict):
        encoder_input_keys = ["raw_feats", "raw_feat_lens", "fc_feats", "attn_feats", "attn_feat_lens"]
        encoder_input = { key: input_dict[key] for key in encoder_input_keys }
        encoder_output_dict = self.encoder(encoder_input)
        if input_dict["mode"] == "train":
            forward_dict = { "mode": "train", "sample_method": "greedy", "temp": 1.0 }
            forward_keys = ["caps", "cap_lens", "ss_ratio", "temporal_label"]
            for key in forward_keys:
                forward_dict[key] = input_dict[key]
            forward_dict.update(encoder_output_dict)
            output = self.train_forward(forward_dict)
        elif input_dict["mode"] == "inference":
            forward_dict = {"mode": "inference"}
            default_args = { "sample_method": "greedy", "max_length": self.max_length, "temp": 1.0 }
            forward_keys = ["sample_method", "max_length", "temp", "temporal_label"]
            for key in forward_keys:
                if key in input_dict:
                    forward_dict[key] = input_dict[key]
                else:
                    forward_dict[key] = default_args[key]
            if forward_dict["sample_method"] == "beam":
                if "beam_size" in input_dict:
                    forward_dict["beam_size"] = input_dict["beam_size"]
                else:
                    forward_dict["beam_size"] = 3
            forward_dict.update(encoder_output_dict)
            output = self.inference_forward(forward_dict)
        else:
            raise Exception("mode should be either 'train' or 'inference'")

        return output

    def seq_forward(self, input_dict):
        caps = input_dict["caps"]
        caps_padding_mask = (caps == self.pad_idx).to(caps.device)
        caps_padding_mask = caps_padding_mask[:, :-1]
        output = self.decoder(
            {
                "word": caps[:, :-1],
                "attn_embs": input_dict["attn_embs"],
                "attn_emb_lens": input_dict["attn_emb_lens"],
                "caps_padding_mask": caps_padding_mask,
                "temporal_label": input_dict["temporal_label"],
            }
        )
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = {
            "attn_embs": input_dict["attn_embs"],
            "attn_emb_lens": input_dict["attn_emb_lens"],
            "temporal_label": input_dict["temporal_label"],
        }
        t = input_dict["t"]
        
        ###############
        # determine input word
        ################
        if input_dict["mode"] == "train" and random.random() < input_dict["ss_ratio"]: # training, scheduled sampling
            word = input_dict["caps"][:, :t+1]
        else:
            start_word = torch.tensor([self.start_idx,] * input_dict["attn_embs"].size(0)).unsqueeze(1).long()
            if t == 0:
                word = start_word
            else:
                word = torch.cat((start_word, output["seqs"][:, :t]), dim=-1)
        # word: [N, T]
        decoder_input["word"] = word

        caps_padding_mask = (word == self.pad_idx).to(input_dict["attn_embs"].device)
        decoder_input["caps_padding_mask"] = caps_padding_mask
        return decoder_input

    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = {}
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]
        ###############
        # prepare attn embeds
        ################
        if t == 0:
            attn_embs = repeat_tensor(input_dict["attn_embs"][i], beam_size)
            attn_emb_lens = repeat_tensor(input_dict["attn_emb_lens"][i], beam_size)
            
            output_i["attn_embs"] = attn_embs
            output_i["attn_emb_lens"] = attn_emb_lens
        decoder_input["attn_embs"] = output_i["attn_embs"]
        decoder_input["attn_emb_lens"] = output_i["attn_emb_lens"]
        ###############
        # prepare temporal_label
        ################
        if t == 0:
            temporal_label = repeat_tensor(input_dict["temporal_label"][i], beam_size)
            output_i["temporal_label"] = temporal_label
        decoder_input["temporal_label"] = output_i["temporal_label"]
        ###############
        # determine input word
        ################
        start_word = torch.tensor([self.start_idx,] * beam_size).unsqueeze(1).long()
        if t == 0:
            word = start_word
        else:
            word = torch.cat((start_word, output_i["seqs"]), dim=-1)
        decoder_input["word"] = word
        caps_padding_mask = (word == self.pad_idx).to(input_dict["attn_embs"].device)
        decoder_input["caps_padding_mask"] = caps_padding_mask

        return decoder_input

class Seq2SeqAttnTemporalModel(CaptionModel):

    def __init__(self, encoder, decoder, **kwargs):    
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                models.decoder.BahAttnDecoder,
                models.decoder.BahAttnDecoder2,
                models.decoder.BahAttnDecoder3,
                models.decoder.BahAttnTemporalDecoder,
            )
        super().__init__(encoder, decoder, **kwargs)

    def forward(self, input_dict):
        """
        input_dict: {
            (required)
            mode: train/inference,
            raw_feats,
            raw_feat_lens,
            fc_feats,
            attn_feats,
            attn_feat_lens,
            [sample_method: greedy],
            [temp: 1.0] (in case of no teacher forcing)

            (optional, mode=train)
            caps,
            cap_lens,
            ss_ratio,

            (optional, mode=inference)
            sample_method: greedy/beam,
            max_length,
            temp,
            beam_size (optional, sample_method=beam),
        }
        """
        encoder_input_keys = ["raw_feats", "raw_feat_lens", "fc_feats", "attn_feats", "attn_feat_lens"]
        encoder_input = { key: input_dict[key] for key in encoder_input_keys }
        encoder_output_dict = self.encoder(encoder_input)
        if input_dict["mode"] == "train":
            forward_dict = { "mode": "train", "sample_method": "greedy", "temp": 1.0 }
            forward_keys = ["caps", "cap_lens", "ss_ratio", "temporal_label"]
            for key in forward_keys:
                forward_dict[key] = input_dict[key]
            forward_dict.update(encoder_output_dict)
            output = self.train_forward(forward_dict)
        elif input_dict["mode"] == "inference":
            forward_dict = {"mode": "inference"}
            default_args = { "sample_method": "greedy", "max_length": self.max_length, "temp": 1.0 }
            forward_keys = ["sample_method", "max_length", "temp", "temporal_label"]
            for key in forward_keys:
                if key in input_dict:
                    forward_dict[key] = input_dict[key]
                else:
                    forward_dict[key] = default_args[key]
            if forward_dict["sample_method"] == "beam":
                if "beam_size" in input_dict:
                    forward_dict["beam_size"] = input_dict["beam_size"]
                else:
                    forward_dict["beam_size"] = 3
            forward_dict.update(encoder_output_dict)
            output = self.inference_forward(forward_dict)
        else:
            raise Exception("mode should be either 'train' or 'inference'")

        return output

    def seq_forward(self, input_dict):
        # Bahdanau attention only supports step-by-step implementation, so we implement forward in 
        # step-by-step manner whether in training or evaluation
        return self.stepwise_forward(input_dict)

    def prepare_output(self, input_dict):
        output = super().prepare_output(input_dict)
        attn_weights = torch.empty(output["seqs"].size(0),
                max(input_dict["attn_emb_lens"]),                    
                output["seqs"].size(1))
        output["attn_weights"] = attn_weights
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = {
            "fc_embs": input_dict["fc_embs"],
            "attn_embs": input_dict["attn_embs"],
            "attn_emb_lens": input_dict["attn_emb_lens"],
            "temporal_label": input_dict["temporal_label"],
            "t": input_dict["t"],
        }
        t = input_dict["t"]
        ###############
        # determine input word
        ################
        if input_dict["mode"] == "train" and random.random() < input_dict["ss_ratio"]: # training, scheduled sampling
            word = input_dict["caps"][:, t]
        else:
            if t == 0:
                word = torch.tensor([self.start_idx,] * input_dict["fc_embs"].size(0)).long()
            else:
                word = output["seqs"][:, t-1]
        # word: [N,]
        decoder_input["word"] = word.unsqueeze(1)

        ################
        # prepare rnn state
        ################
        if t > 0:
            decoder_input["state"] = output["state"]
        return decoder_input

    def stepwise_process_step(self, output, output_t):
        super().stepwise_process_step(output, output_t)
        output["state"] = output_t["state"]
        t = output_t["t"]
        output["attn_weights"][:, :, t] = output_t["weights"]

    def prepare_beamsearch_output(self, input_dict):
        output = super().prepare_beamsearch_output(input_dict)
        beam_size = input_dict["beam_size"]
        max_length = input_dict["max_length"]
        output["attn_weights"] = torch.empty(beam_size,
                                             max(input_dict["attn_emb_lens"]),
                                             max_length)
        return output


    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = {"t": input_dict['t']}
        
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]
        ###############
        # prepare fc embeds
        ################
        if t == 0:
            fc_embs = repeat_tensor(input_dict["fc_embs"][i], beam_size)
            output_i["fc_embs"] = fc_embs
        decoder_input["fc_embs"] = output_i["fc_embs"]

        if t == 0:
            temporal_label = repeat_tensor(input_dict["temporal_label"][i], beam_size)
            output_i["temporal_label"] = temporal_label
        decoder_input["temporal_label"] = output_i["temporal_label"]
        ###############
        # prepare attn embeds
        ################
        if t == 0:
            attn_embs = repeat_tensor(input_dict["attn_embs"][i], beam_size)
            attn_emb_lens = repeat_tensor(input_dict["attn_emb_lens"][i], beam_size)
            output_i["attn_embs"] = attn_embs
            output_i["attn_emb_lens"] = attn_emb_lens
        decoder_input["attn_embs"] = output_i["attn_embs"]
        decoder_input["attn_emb_lens"] = output_i["attn_emb_lens"]

         
          
        ###############
        # determine input word
        ################
        if t == 0:
            word = torch.tensor([self.start_idx,] * beam_size).long()
        else:
            word = output_i["next_word"]
        decoder_input["word"] = word.unsqueeze(1)

        ################
        # prepare rnn state
        ################
        if t > 0:
            if self.decoder.rnn_type == "LSTM":
                decoder_input["state"] = (output_i["state"][0][:, output_i["prev_words_beam"], :].contiguous(),
                                          output_i["state"][1][:, output_i["prev_words_beam"], :].contiguous())
            else:
                decoder_input["state"] = output_i["state"][:, output_i["prev_words_beam"], :].contiguous()

        return decoder_input

    def beamsearch_process_step(self, output_i, output_t):
        t = output_t["t"]
        output_i["state"] = output_t["state"]
        output_i["attn_weights"][..., t] = output_t["weights"]
        output_i["attn_weights"] = output_i["attn_weights"][output_i["prev_words_beam"], ...]

    def beamsearch_process(self, output, output_i, input_dict):
        super().beamsearch_process(output, output_i, input_dict)
        i = input_dict["sample_idx"]
        output["attn_weights"][i] = output_i["attn_weights"][0]
        

class Seq2SeqAttnSedprobModel(CaptionModel):

    def __init__(self, encoder, decoder, **kwargs):    
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                models.decoder.BahAttnDecoder,
                models.decoder.BahAttnDecoder2,
                models.decoder.BahAttnDecoder3,
                models.decoder.BahAttnSedprobDecoder,
                models.decoder.BahAttnSedprobAttnDecoder,
            )
        super().__init__(encoder, decoder, **kwargs)
        self.sed_prob_pool = torch.nn.AvgPool1d(8, 8)

    def forward(self, input_dict):
        """
        input_dict: {
            (required)
            mode: train/inference,
            raw_feats,
            raw_feat_lens,
            fc_feats,
            attn_feats,
            attn_feat_lens,
            [sample_method: greedy],
            [temp: 1.0] (in case of no teacher forcing)

            (optional, mode=train)
            caps,
            cap_lens,
            ss_ratio,

            (optional, mode=inference)
            sample_method: greedy/beam,
            max_length,
            temp,
            beam_size (optional, sample_method=beam),
        }
        """
        encoder_input_keys = ["raw_feats", "raw_feat_lens", "fc_feats", "attn_feats", "attn_feat_lens"]
        encoder_input = { key: input_dict[key] for key in encoder_input_keys }
        encoder_output_dict = self.encoder(encoder_input)
        e_s = encoder_output_dict["attn_embs"].shape
        if input_dict["sed_prob"].shape[1] >= encoder_output_dict["attn_embs"].shape[1] * 8: #cnn 8 
            sed_prob_stemmed = input_dict["sed_prob"][:, :e_s[1]*8, :].reshape(e_s[0], e_s[1], 8, 447)
            #input_dict["sed_prob"] = self.sed_prob_pool(sed_prob_stemmed.permute(0, 2, 1)).permute(0, 2, 1)
            input_dict["sed_prob"] = linear_softmax_pool_with_lens(sed_prob_stemmed, encoder_output_dict["attn_emb_lens"])
        else: # cnn14
        #elif input_dict["sed_prob"].shape[1] != encoder_output_dict["attn_embs"].shape[1]: #cnn 14 inference
            input_dict["sed_prob"] = input_dict["sed_prob"][:, :e_s[1], :]
        if input_dict["mode"] == "train":
            forward_dict = { "mode": "train", "sample_method": "greedy", "temp": 1.0 }
            forward_keys = ["caps", "cap_lens", "ss_ratio", "sed_prob"]
            for key in forward_keys:
                forward_dict[key] = input_dict[key]
            forward_dict.update(encoder_output_dict)
            output = self.train_forward(forward_dict)
        elif input_dict["mode"] == "inference":
            forward_dict = {"mode": "inference"}
            default_args = { "sample_method": "greedy", "max_length": self.max_length, "temp": 1.0 }
            forward_keys = ["sample_method", "max_length", "temp", "sed_prob"]
            for key in forward_keys:
                if key in input_dict:
                    forward_dict[key] = input_dict[key]
                else:
                    forward_dict[key] = default_args[key]
            if forward_dict["sample_method"] == "beam":
                if "beam_size" in input_dict:
                    forward_dict["beam_size"] = input_dict["beam_size"]
                else:
                    forward_dict["beam_size"] = 3
            forward_dict.update(encoder_output_dict)
            output = self.inference_forward(forward_dict)
        else:
            raise Exception("mode should be either 'train' or 'inference'")

        return output

    def seq_forward(self, input_dict):
        # Bahdanau attention only supports step-by-step implementation, so we implement forward in 
        # step-by-step manner whether in training or evaluation
        return self.stepwise_forward(input_dict)

    def prepare_output(self, input_dict):
        output = super().prepare_output(input_dict)
        attn_weights = torch.empty(output["seqs"].size(0),
                max(input_dict["attn_emb_lens"]),                    
                output["seqs"].size(1))
        output["attn_weights"] = attn_weights
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = {
            "fc_embs": input_dict["fc_embs"],
            "attn_embs": input_dict["attn_embs"],
            "attn_emb_lens": input_dict["attn_emb_lens"],
            "sed_prob": input_dict["sed_prob"],
            "t": input_dict["t"],
        }
        t = input_dict["t"]
        ###############
        # determine input word
        ################
        if input_dict["mode"] == "train" and random.random() < input_dict["ss_ratio"]: # training, scheduled sampling
            word = input_dict["caps"][:, t]
        else:
            if t == 0:
                word = torch.tensor([self.start_idx,] * input_dict["fc_embs"].size(0)).long()
            else:
                word = output["seqs"][:, t-1]
        # word: [N,]
        decoder_input["word"] = word.unsqueeze(1)

        ################
        # prepare rnn state
        ################
        if t > 0:
            decoder_input["state"] = output["state"]
        return decoder_input

    def stepwise_process_step(self, output, output_t):
        super().stepwise_process_step(output, output_t)
        output["state"] = output_t["state"]
        t = output_t["t"]
        output["attn_weights"][:, :, t] = output_t["weights"]

    def prepare_beamsearch_output(self, input_dict):
        output = super().prepare_beamsearch_output(input_dict)
        beam_size = input_dict["beam_size"]
        max_length = input_dict["max_length"]
        output["attn_weights"] = torch.empty(beam_size,
                                             max(input_dict["attn_emb_lens"]),
                                             max_length)
        return output


    def prepare_beamsearch_decoder_input(self, input_dict, output_i):
        decoder_input = {"t": input_dict['t']}
        
        t = input_dict["t"]
        i = input_dict["sample_idx"]
        beam_size = input_dict["beam_size"]

        repeat_keys = ["fc_embs", "sed_prob", "attn_embs", "attn_emb_lens"]
        for key in repeat_keys:
            if t == 0:
                output_i[key] = repeat_tensor(input_dict[key][i], beam_size)
            decoder_input[key] = output_i[key]
        """
        ###############
        # prepare fc embeds
        ################
        if t == 0:
            fc_embs = repeat_tensor(input_dict["fc_embs"][i], beam_size)
            output_i["fc_embs"] = fc_embs
        decoder_input["fc_embs"] = output_i["fc_embs"]

        if t == 0:
            sed_prob  = repeat_tensor(input_dict["sed_prob"][i], beam_size)
            output_i["sed_prob"] = sed_prob
        decoder_input["sed_prob"] = output_i["sed_prob"]
        ###############
        # prepare attn embeds
        ################
        if t == 0:
            attn_embs = repeat_tensor(input_dict["attn_embs"][i], beam_size)
            attn_emb_lens = repeat_tensor(input_dict["attn_emb_lens"][i], beam_size)
            output_i["attn_embs"] = attn_embs
            output_i["attn_emb_lens"] = attn_emb_lens
        decoder_input["attn_embs"] = output_i["attn_embs"]
        decoder_input["attn_emb_lens"] = output_i["attn_emb_lens"]
        """
         
          
        ###############
        # determine input word
        ################
        if t == 0:
            word = torch.tensor([self.start_idx,] * beam_size).long()
        else:
            word = output_i["next_word"]
        decoder_input["word"] = word.unsqueeze(1)

        ################
        # prepare rnn state
        ################
        if t > 0:
            if self.decoder.rnn_type == "LSTM":
                decoder_input["state"] = (output_i["state"][0][:, output_i["prev_words_beam"], :].contiguous(),
                                          output_i["state"][1][:, output_i["prev_words_beam"], :].contiguous())
            else:
                decoder_input["state"] = output_i["state"][:, output_i["prev_words_beam"], :].contiguous()

        return decoder_input

    def beamsearch_process_step(self, output_i, output_t):
        t = output_t["t"]
        output_i["state"] = output_t["state"]
        output_i["attn_weights"][..., t] = output_t["weights"]
        output_i["attn_weights"] = output_i["attn_weights"][output_i["prev_words_beam"], ...]

    def beamsearch_process(self, output, output_i, input_dict):
        super().beamsearch_process(output, output_i, input_dict)
        i = input_dict["sample_idx"]
        output["attn_weights"][i] = output_i["attn_weights"][0]

