# -*- coding: utf-8 -*-
import random
import torch
import torch.nn as nn

from models.base_model import CaptionModel
from models.utils import repeat_tensor
import models.decoder


class TransformerModel(CaptionModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                models.decoder.TransformerDecoder
            )
        super().__init__(encoder, decoder, **kwargs)

    def seq_forward(self, input_dict):
        caps = input_dict["caps"]
        caps_padding_mask = (caps == self.pad_idx).to(caps.device)
        caps_padding_mask = caps_padding_mask[:, :-1]
        output = self.decoder(
            {
                "word": caps[:, :-1],
                "attn_embs": input_dict["attn_embs"],
                "attn_emb_lens": input_dict["attn_emb_lens"],
                "caps_padding_mask": caps_padding_mask
            }
        )
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = {
            "attn_embs": input_dict["attn_embs"],
            "attn_emb_lens": input_dict["attn_emb_lens"]
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

class TransformerAddinfoModel(TransformerModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.decoder.TransformerAddinfoDecoder
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
            add_info,
            add_info_lens,
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
            forward_keys = ["caps", "cap_lens", "ss_ratio", "add_info", "add_info_lens"]
            for key in forward_keys:
                forward_dict[key] = input_dict[key]
            forward_dict.update(encoder_output_dict)
            output = self.train_forward(forward_dict)
        elif input_dict["mode"] == "inference":
            forward_dict = {"mode": "inference"}
            default_args = { "sample_method": "greedy", "max_length": self.max_length, "temp": 1.0 }
            forward_keys = ["sample_method", "max_length", "temp", "add_info", "add_info_lens"]
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
                "add_info": input_dict["add_info"],
                "add_info_lens": input_dict["add_info_lens"],
            }
        )
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = {
            "attn_embs": input_dict["attn_embs"],
            "attn_emb_lens": input_dict["attn_emb_lens"],
            "add_info": input_dict["add_info"],
            "add_info_lens": input_dict["add_info_lens"],
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
            add_info = repeat_tensor(input_dict["add_info"][i], beam_size)
            add_info_lens = repeat_tensor(input_dict["add_info_lens"][i], beam_size)
            output_i["attn_embs"] = attn_embs
            output_i["attn_emb_lens"] = attn_emb_lens
            output_i["add_info"] = add_info
            output_i["add_info_lens"] = add_info_lens
        decoder_input["attn_embs"] = output_i["attn_embs"]
        decoder_input["attn_emb_lens"] = output_i["attn_emb_lens"]
        decoder_input["add_info"] = output_i["add_info"]
        decoder_input["add_info_lens"] = output_i["add_info_lens"]
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

class M2TransformerModel(CaptionModel):

    def __init__(self, encoder: nn.Module, decoder: nn.Module, **kwargs):
        if not hasattr(self, "compatible_decoders"):
            self.compatible_decoders = (
                captioning.models.decoder.M2TransformerDecoder
            )
        super().__init__(encoder, decoder, **kwargs)
        self.check_encoder_compatibility()

    def check_encoder_compatibility(self):
        assert isinstance(self.encoder, captioning.models.encoder.M2TransformerEncoder), \
            f"only M2TransformerModel is compatible with {self.__class__.__name__}"


    def seq_forward(self, input_dict):
        caps = input_dict["caps"]
        output = self.decoder(
            {
                "word": caps[:, :-1],
                "attn_embs": input_dict["attn_embs"],
                "attn_emb_mask": input_dict["attn_emb_mask"],
            }
        )
        return output

    def prepare_decoder_input(self, input_dict, output):
        decoder_input = {
            "attn_embs": input_dict["attn_embs"],
            "attn_emb_mask": input_dict["attn_emb_mask"]
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
            attn_emb_mask = repeat_tensor(input_dict["attn_emb_mask"][i], beam_size)
            output_i["attn_embs"] = attn_embs
            output_i["attn_emb_mask"] = attn_emb_mask
        decoder_input["attn_embs"] = output_i["attn_embs"]
        decoder_input["attn_emb_mask"] = output_i["attn_emb_mask"]
        ###############
        # determine input word
        ################
        start_word = torch.tensor([self.start_idx,] * beam_size).unsqueeze(1).long()
        if t == 0:
            word = start_word
        else:
            word = torch.cat((start_word, output_i["seqs"]), dim=-1)
        decoder_input["word"] = word

        return decoder_input
